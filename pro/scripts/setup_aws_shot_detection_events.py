#!/usr/bin/env python3
"""Create SNS/SQS/IAM resources for AWS Rekognition shot profile events."""

from __future__ import annotations

import argparse
import json

import boto3
from botocore.exceptions import ClientError


def ensure_role(role_name: str, account_id: str) -> str:
    iam = boto3.client("iam")
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "rekognition.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        role = iam.get_role(RoleName=role_name)["Role"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
        role = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust))["Role"]

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sns:Publish",
                "Resource": f"arn:aws:sns:*:{account_id}:skyflow-shot-detection-*",
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="skyflow-rekognition-sns-publish",
        PolicyDocument=json.dumps(policy),
    )
    return role["Arn"]


def ensure_region_resources(region: str) -> dict[str, str]:
    sns = boto3.client("sns", region_name=region)
    sqs = boto3.client("sqs", region_name=region)
    topic_arn = sns.create_topic(Name=f"skyflow-shot-detection-{region}")["TopicArn"]
    queue_url = sqs.create_queue(
        QueueName=f"skyflow-shot-detection-{region}",
        Attributes={"VisibilityTimeout": "120", "MessageRetentionPeriod": "86400"},
    )["QueueUrl"]
    queue_arn = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
    queue_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowSnsSendMessage",
                "Effect": "Allow",
                "Principal": {"Service": "sns.amazonaws.com"},
                "Action": "sqs:SendMessage",
                "Resource": queue_arn,
                "Condition": {"ArnEquals": {"aws:SourceArn": topic_arn}},
            }
        ],
    }
    sqs.set_queue_attributes(QueueUrl=queue_url, Attributes={"Policy": json.dumps(queue_policy)})
    sns.subscribe(TopicArn=topic_arn, Protocol="sqs", Endpoint=queue_arn)
    return {"region": region, "topic_arn": topic_arn, "queue_url": queue_url, "queue_arn": queue_arn}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-name", default="skyflow-rekognition-sns-role")
    parser.add_argument("--regions", default="us-west-2,us-east-2,ap-southeast-1,eu-central-1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    regions = [region.strip() for region in args.regions.split(",") if region.strip()]
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = ensure_role(args.role_name, account_id)
    print(json.dumps({"role_arn": role_arn}))
    for region in regions:
        print(json.dumps(ensure_region_resources(region)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
