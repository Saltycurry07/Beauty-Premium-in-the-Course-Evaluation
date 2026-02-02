#!/usr/bin/env python3
"""Fetch professor ratings for selected schools/departments from RateMyProfessors."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable

GRAPHQL_ENDPOINT = "https://www.ratemyprofessors.com/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; STATS201-crawler/1.0)",
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

TARGET_SCHOOLS = {
    "UCLA": "University of California Los Angeles (UCLA)",
    "NYU": "New York University",
}
TARGET_DEPARTMENTS = ["Computer Science", "Fine Arts"]


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
    return " ".join(text.strip().lower().split())


def find_school_id(client: GraphQLClient, school_name: str) -> tuple[str, str]:
    variables = {"query": {"text": school_name}, "first": 10}
    data = client.post(SCHOOL_QUERY, variables)
    edges = data["data"]["newSearch"]["schools"]["edges"]
    for edge in edges:
        node = edge["node"]
        if normalize(node["name"]) == normalize(school_name):
            return node["id"], node["name"]
    for edge in edges:
        node = edge["node"]
        if normalize(school_name) in normalize(node["name"]):
            return node["id"], node["name"]
    raise RuntimeError(f"School not found for query: {school_name}")


def fetch_department_id(client: GraphQLClient, school_id: str, department_name: str) -> str:
    variables = {
        "query": {"text": "", "schoolID": school_id, "fallback": True},
        "first": 1,
        "after": None,
    }
    data = client.post(TEACHER_QUERY, variables)
    filters = data["data"]["newSearch"]["teachers"]["filters"]
    for filter_item in filters:
        if filter_item["field"] == "teacherdepartment_s":
            for option in filter_item["options"]:
                if not option["id"]:
                    continue
                if normalize(option["value"]) == normalize(department_name):
                    return option["id"]
    available = []
    for filter_item in filters:
        if filter_item["field"] == "teacherdepartment_s":
            available = [opt["value"] for opt in filter_item["options"] if opt["id"]]
    raise RuntimeError(
        f"Department '{department_name}' not found for school {school_id}. "
        f"Available examples: {available[:10]}"
    )


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
                department=node.get("department") or "",
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
        for record in records:
            writer.writerow(
                [
                    record.school,
                    record.department,
                    record.professor,
                    record.avg_rating,
                    record.avg_difficulty,
                    record.would_take_again_percent,
                ]
            )


def build_records() -> list[TeacherRecord]:
    client = GraphQLClient(GRAPHQL_ENDPOINT, HEADERS)
    all_records: list[TeacherRecord] = []
    for label, school_name in TARGET_SCHOOLS.items():
        school_id, resolved_name = find_school_id(client, school_name)
        for department_name in TARGET_DEPARTMENTS:
            department_id = fetch_department_id(client, school_id, department_name)
            records = list(iter_teachers(client, school_id, department_id))
            if not records:
                print(
                    f"Warning: no results for {label} {department_name}",
                    file=sys.stderr,
                )
            all_records.extend(records)
            print(
                f"Fetched {len(records)} professors for {resolved_name} / {department_name}",
                file=sys.stderr,
            )
    return all_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch RateMyProfessors ratings for UCLA/NYU Computer Science and Fine Arts "
            "faculty and output a CSV file."
        )
    )
    parser.add_argument(
        "--output",
        default="rmp_ucla_nyu_professors.csv",
        help="Path to write the CSV file (default: rmp_ucla_nyu_professors.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records()
    if not records:
        raise SystemExit("No records returned from RateMyProfessors.")
    write_csv(records, args.output)
    print(f"Saved {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
