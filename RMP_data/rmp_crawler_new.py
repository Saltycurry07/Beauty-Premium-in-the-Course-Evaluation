#!/usr/bin/env python3
"""
Fetch professor ratings for selected schools/departments from RateMyProfessors (RMP).
Uses GraphQL search; resolves each department to a departmentID using RMP's own filter options.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional

GRAPHQL_ENDPOINT = "https://www.ratemyprofessors.com/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; stats201-crawler/1.0)",
    "Referer": "https://www.ratemyprofessors.com/",
}

SCHOOL_QUERY = """
query SearchSchools($query: SchoolSearchQuery!, $first: Int) {
  newSearch {
    schools(query: $query, first: $first) {
      edges {
        node {
          id
          name
          city
          state
        }
      }
    }
  }
}
"""

TEACHER_QUERY = """
query SearchTeachers($query: TeacherSearchQuery!, $first: Int, $after: String) {
  newSearch {
    teachers(query: $query, first: $first, after: $after) {
      filters {
        field
        options {
          id
          value
          count
        }
      }
      edges {
        node {
          id
          firstName
          lastName
          department
          avgRatingRounded
          avgDifficultyRounded
          wouldTakeAgainPercentRounded
          school {
            name
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
      resultCount
    }
  }
}
"""

# -------------------------
# 1) Schools you target
# -------------------------
TARGET_SCHOOLS = {
    "MIT":  "Massachusetts Institute of Technology",
    "UCLA": "University of California Los Angeles (UCLA)",
    "USC":  "University of Southern California",
    "UIUC": "University of Illinois at Urbana - Champaign",
    "OSU":  "Ohio State University",
    "WFU":  "Wake Forest University",
}

# -------------------------
# 2) Department lists (by school label)
#    Edit here only, everything else should work.
# -------------------------
TARGET_DEPARTMENTS_BY_SCHOOL = {
    # MIT (as provided)
    "MIT": [
        "Aeronautics & Astronautics",
        "Biological Engineering",
        "Chemical Engineering",
        "Civil & Environmental Engineering",
        "Electrical Engineering & Computer Science",
        "Materials Science & Engineering",
        "Mechanical Engineering",
        "Nuclear Science & Engineering",
        "Institute for Data, Systems & Society",
        "Institute for Medical Engineering & Science",
    ],

    # UCLA Samueli (as provided)
    "UCLA": [
        "Bioengineering",
        "Chemical and Biomolecular Engineering",
        "Civil and Environmental Engineering",
        "Computer Science",
        "Electrical and Computer Engineering",
        "Mechanical and Aerospace Engineering",
        "Materials Science and Engineering",
    ],

    # USC Viterbi (as provided)
    "USC": [
        "Aerospace and Mechanical Engineering",
        "Astronautical Engineering",
        "Biomedical Engineering",
        "Chemical Engineering and Materials Science",
        "Civil and Environmental Engineering",
        "Computer Science",
        "Electrical and Computer Engineering",
        "Engineering in Society Program",
        "Industrial and Systems Engineering",
        "Technology and Applied Computing Program",
    ],

    # UIUC Grainger (as provided)
    "UIUC": [
        "Aerospace Engineering",
        "Agricultural & Biological Engineering",
        "Bioengineering",
        "Chemical & Biomolecular Engineering",
        "Civil & Environmental Engineering",
        "Electrical & Computer Engineering",
        "Industrial & Enterprise Systems Engineering",
        "Materials Science & Engineering",
        "Mechanical Science & Engineering",
        "Nuclear, Plasma & Radiological Engineering",
        "Physics",
        "Siebel School of Computing and Data Science",
    ],

    # OSU College of Engineering (as provided)
    "OSU": [
        "Biomedical Engineering",
        "Center for Aviation Studies",
        "Chemical Engineering",
        "Civil, Environmental and Geodetic Engineering",
        "Computer Science and Engineering",
        "Electrical and Computer Engineering",
        "Engineering Education",
        "Food, Agricultural and Biological Engineering",
        "Integrated Systems Engineering",
        "Knowlton School of Architecture",
        "Materials Science and Engineering",
        "Mechanical and Aerospace Engineering",
    ],

    # Wake Forest: single Engineering Department (as you said)
    "WFU": [
        "Engineering",
        "Engineering Department",
    ],
}


@dataclass
class TeacherRecord:
    school: str
    department: str
    professor: str
    avg_rating: float | None
    avg_difficulty: float | None
    would_take_again_percent: float | None


class GraphQLClient:
    def __init__(self, endpoint: str, headers: dict[str, str]) -> None:
        self.endpoint = endpoint
        self.headers = headers

    def post(self, query: str, variables: dict) -> dict:
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=payload, headers=self.headers)
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
        if data.get("errors"):
            message = data["errors"][0].get("message", "Unknown GraphQL error")
            raise RuntimeError(message)
        return data


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def norm_loose(text: str) -> str:
    """
    Looser normalization for matching department names:
    - lower
    - replace & -> and
    - drop punctuation
    - collapse whitespace
    """
    t = (text or "").lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = " ".join(t.split())
    return t


def find_school_id(client: GraphQLClient, school_name: str) -> tuple[str, str]:
    variables = {"query": {"text": school_name}, "first": 10}
    data = client.post(SCHOOL_QUERY, variables)
    edges = data["data"]["newSearch"]["schools"]["edges"]

    # exact normalized match
    for edge in edges:
        node = edge["node"]
        if normalize(node["name"]) == normalize(school_name):
            return node["id"], node["name"]

    # substring match
    for edge in edges:
        node = edge["node"]
        if normalize(school_name) in normalize(node["name"]):
            return node["id"], node["name"]

    candidates = [edge["node"]["name"] for edge in edges]
    raise RuntimeError(
        f"School not found for query: {school_name}. Top candidates: {candidates[:10]}"
    )


def get_department_options(client: GraphQLClient, school_id: str) -> list[dict]:
    """
    Pull the department filter options for a school by making a minimal teacher query.
    """
    variables = {
        "query": {"text": "", "schoolID": school_id, "fallback": True},
        "first": 1,
        "after": None,
    }
    data = client.post(TEACHER_QUERY, variables)
    filters = data["data"]["newSearch"]["teachers"]["filters"]
    for f in filters:
        if f["field"] == "teacherdepartment_s":
            return [opt for opt in f["options"] if opt.get("id")]
    return []


def resolve_department_id(
    client: GraphQLClient,
    school_id: str,
    department_name: str,
) -> Optional[str]:
    """
    Resolve department_name to RMP's departmentID.
    Tries:
      1) exact normalize(value) match
      2) loose normalization match (and/& punctuation-insensitive)
      3) substring match (either direction) on loose strings
    Returns departmentID or None.
    """
    opts = get_department_options(client, school_id)
    if not opts:
        return None

    target_exact = normalize(department_name)
    target_loose = norm_loose(department_name)

    # 1) exact normalize match
    for opt in opts:
        if normalize(opt["value"]) == target_exact:
            return opt["id"]

    # 2) loose match
    for opt in opts:
        if norm_loose(opt["value"]) == target_loose:
            return opt["id"]

    # 3) substring match (loose)
    for opt in opts:
        v = norm_loose(opt["value"])
        if target_loose and (target_loose in v or v in target_loose):
            return opt["id"]

    return None


def iter_teachers(
    client: GraphQLClient, school_id: str, department_id: str
) -> Iterable[TeacherRecord]:
    after = None
    while True:
        variables = {
            "query": {
                "text": "",
                "schoolID": school_id,
                "departmentID": department_id,
                "fallback": True,
            },
            "first": 100,
            "after": after,
        }
        data = client.post(TEACHER_QUERY, variables)
        teachers = data["data"]["newSearch"]["teachers"]

        for edge in teachers["edges"]:
            node = edge["node"]
            yield TeacherRecord(
                school=node["school"]["name"],
                department=(node.get("department") or "").strip(),
                professor=f"{node.get('firstName', '').strip()} {node.get('lastName', '').strip()}".strip(),
                avg_rating=node.get("avgRatingRounded"),
                avg_difficulty=node.get("avgDifficultyRounded"),
                would_take_again_percent=node.get("wouldTakeAgainPercentRounded"),
            )

        page_info = teachers["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        after = page_info["endCursor"]
        time.sleep(0.2)


def write_csv(records: list[TeacherRecord], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "school",
                "department",
                "professor",
                "avg_rating",
                "avg_difficulty",
                "would_take_again_percent",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.school,
                    r.department,
                    r.professor,
                    r.avg_rating,
                    r.avg_difficulty,
                    r.would_take_again_percent,
                ]
            )


def build_records() -> list[TeacherRecord]:
    client = GraphQLClient(GRAPHQL_ENDPOINT, HEADERS)
    all_records: list[TeacherRecord] = []

    for label, school_name in TARGET_SCHOOLS.items():
        school_id, resolved_school_name = find_school_id(client, school_name)

        dept_list = TARGET_DEPARTMENTS_BY_SCHOOL.get(label, [])
        if not dept_list:
            print(f"Warning: no department list for {label}. Skipping.", file=sys.stderr)
            continue

        # Cache department options once for debug messages
        dept_options = get_department_options(client, school_id)
        dept_option_values = [opt["value"] for opt in dept_options][:30]

        for dept_name in dept_list:
            dept_id = resolve_department_id(client, school_id, dept_name)
            if not dept_id:
                print(
                    f"[WARN] Department not found on RMP: {resolved_school_name} / '{dept_name}'. "
                    f"Examples of available departments: {dept_option_values}",
                    file=sys.stderr,
                )
                continue

            records = list(iter_teachers(client, school_id, dept_id))
            if not records:
                print(
                    f"[WARN] No teachers returned: {resolved_school_name} / '{dept_name}'",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Fetched {len(records)} teachers for {resolved_school_name} / {dept_name}",
                    file=sys.stderr,
                )
            all_records.extend(records)

    return all_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch RateMyProfessors ratings for selected schools and department lists; output a CSV file."
    )
    parser.add_argument(
        "--output",
        default="rmp_engineering_departments.csv",
        help="Path to write the CSV file (default: rmp_engineering_departments.csv)",
    )
    args, _ = parser.parse_known_args()  # <-- key change: ignore unknown args like -f
    return args


def main() -> None:
    args = parse_args()
    records = build_records()
    if not records:
        raise SystemExit("No records returned from RateMyProfessors.")
    write_csv(records, args.output)
    print(f"Saved {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
