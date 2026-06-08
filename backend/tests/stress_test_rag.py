"""RAG Pipeline Stress Test Suite

Creates synthetic test documents with known facts, uploads them,
queries the RAG system, and evaluates answer quality.
"""

import json
import time
import httpx
import sys
from pathlib import Path
from dataclasses import dataclass, field

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 120.0

# ── Test Documents ──────────────────────────────────────────────────

DOC_COMPANY_POLICY = """
ACME Corporation Employee Handbook - 2024 Edition

Chapter 1: Leave Policy

1.1 Annual Leave
All full-time employees are entitled to 24 days of paid annual leave per calendar year.
Part-time employees receive leave on a pro-rata basis. Leave cannot be carried over beyond March 31 of the following year.
Maximum carry-over is capped at 5 days.

1.2 Sick Leave
Employees are entitled to 12 days of paid sick leave per year. A medical certificate is required for absences exceeding 3 consecutive working days. Unused sick leave cannot be encashed or carried over.

1.3 Maternity Leave
Female employees are entitled to 26 weeks of paid maternity leave for the first two children.
For the third child onwards, entitlement is 12 weeks.

1.4 Paternity Leave
Male employees are entitled to 15 working days of paternity leave within 6 months of the child's birth.

Chapter 2: Compensation

2.1 Salary Structure
The salary structure consists of the following components:
- Basic Salary: 40% of CTC
- House Rent Allowance (HRA): 20% of CTC
- Special Allowance: 25% of CTC
- Employer PF Contribution: 12% of Basic Salary
- Gratuity: 4.81% of Basic Salary

2.2 Performance Bonus
Annual performance bonus ranges from 0% to 25% of annual CTC, based on individual performance rating:
- Rating 5 (Exceptional): 20-25% bonus
- Rating 4 (Exceeds Expectations): 15-20% bonus
- Rating 3 (Meets Expectations): 8-12% bonus
- Rating 2 (Below Expectations): 0-5% bonus
- Rating 1 (Unsatisfactory): No bonus

2.3 Salary Revision
Annual salary revisions take effect from April 1st each year. The average company-wide increment for FY2024 was 9.2%.

Chapter 3: Work Hours and Remote Work

3.1 Standard Work Hours
Standard working hours are 9:00 AM to 6:00 PM, Monday through Friday.
Core hours when all employees must be available are 10:30 AM to 4:00 PM.
Flexible start time is between 8:00 AM and 10:30 AM.

3.2 Remote Work Policy
Employees may work remotely up to 3 days per week with manager approval.
Fully remote arrangements require VP-level approval.
All remote workers must be available during core hours in their local timezone.
Remote work equipment allowance is INR 25,000 (one-time) and INR 1,500/month for internet reimbursement.

Chapter 4: Travel and Expense Policy

4.1 Domestic Travel
- Air travel: Economy class for distances over 500 km
- Hotel: Up to INR 5,000 per night for non-metro cities, INR 8,000 for metro cities
- Daily meal allowance: INR 1,500 for domestic travel
- Local transport: Actual expenses with receipts

4.2 International Travel
- Air travel: Economy class; Business class for flights over 8 hours with VP approval
- Hotel: Up to USD 200 per night; USD 300 in high-cost cities (NYC, London, Tokyo, Singapore)
- Daily meal allowance: USD 75
- Visa processing: Fully reimbursed

Chapter 5: Code of Conduct

5.1 Anti-Harassment Policy
ACME Corporation maintains a zero-tolerance policy toward harassment of any kind.
All complaints must be reported to the Internal Complaints Committee (ICC) within 90 days of the incident.
The ICC must complete its investigation within 60 days of receiving a complaint.
Penalties range from written warnings to immediate termination depending on severity.

5.2 Conflict of Interest
Employees must declare any potential conflicts of interest annually.
Employment with competitors or clients is strictly prohibited during the tenure of employment.
Moonlighting is permitted only with written HR approval and must not conflict with ACME's business interests.

5.3 Data Security
All company data is classified into three tiers:
- Tier 1 (Confidential): Financial records, client data, trade secrets. Access restricted to authorized personnel only.
- Tier 2 (Internal): Internal communications, project documents. Accessible to all employees.
- Tier 3 (Public): Marketing materials, press releases. No restrictions.

Violation of data security policies may result in termination and legal action.
"""

DOC_TECHNICAL_SPEC = """
Project Aurora - Technical Architecture Document v2.3

1. System Overview

Project Aurora is a distributed real-time data processing platform designed to handle up to 500,000 events per second with a p99 latency of under 50 milliseconds. The system is deployed across 3 AWS regions: us-east-1, eu-west-1, and ap-southeast-1.

2. Architecture Components

2.1 Ingestion Layer
- Apache Kafka clusters with 12 brokers per region
- Topic partitioning: 64 partitions per topic
- Replication factor: 3
- Message retention: 7 days
- Average message size: 2.4 KB
- Daily ingestion volume: approximately 43 TB

2.2 Processing Layer
- Apache Flink cluster with 48 task managers
- Parallelism: 128 per job
- Checkpoint interval: 60 seconds
- State backend: RocksDB with incremental checkpoints
- Processing guarantees: Exactly-once semantics

2.3 Storage Layer
- Primary store: Apache Cassandra cluster with 24 nodes, RF=3
- Analytical store: ClickHouse cluster with 8 shards, 3 replicas each
- Cache layer: Redis Cluster with 6 masters and 6 replicas
- Object storage: S3 with lifecycle policy (hot: 30 days, warm: 90 days, glacier: 1 year)

2.4 API Gateway
- Kong API Gateway
- Rate limiting: 10,000 requests per minute per API key
- Authentication: OAuth 2.0 with JWT tokens
- API versioning: URI-based (v1, v2)
- Current active API version: v2.3

3. Performance Benchmarks

3.1 Latency
- p50: 12ms
- p90: 28ms
- p99: 47ms
- p99.9: 89ms

3.2 Throughput
- Sustained: 500,000 events/second
- Peak tested: 1,200,000 events/second
- Error rate at peak: 0.003%

3.3 Availability
- SLA target: 99.95%
- Achieved (last 12 months): 99.97%
- Planned maintenance windows: Monthly, 2nd Sunday, 2:00-4:00 AM UTC

4. Security Architecture

4.1 Encryption
- Data at rest: AES-256
- Data in transit: TLS 1.3
- Key management: AWS KMS with annual rotation
- Database encryption: Transparent Data Encryption (TDE)

4.2 Network Security
- VPC peering between regions
- Private subnets for all data stores
- WAF rules: 247 active rules
- DDoS protection: AWS Shield Advanced

5. Monitoring Stack

5.1 Observability
- Metrics: Prometheus + Grafana (15-second scrape interval)
- Logs: ELK stack (Elasticsearch, Logstash, Kibana)
- Traces: Jaeger with OpenTelemetry SDK
- Alerting: PagerDuty integration with 4-tier escalation policy

5.2 SLI/SLO Dashboard
- Error budget remaining: 78.4% (as of March 2024)
- Burn rate alert threshold: 2x normal over 1 hour

6. Disaster Recovery

6.1 RPO/RTO
- Recovery Point Objective (RPO): 5 minutes
- Recovery Time Objective (RTO): 30 minutes
- Backup frequency: Continuous replication + daily snapshots

6.2 Failover Procedure
- Automatic failover for Kafka and Cassandra
- Manual failover for Flink jobs (estimated 15 minutes)
- DNS failover via Route53 health checks (TTL: 60 seconds)
"""

DOC_FINANCIAL_REPORT = """
Quarterly Financial Report - Q3 FY2024 (October - December 2023)
TechVista Solutions Private Limited

Executive Summary

TechVista Solutions reported strong Q3 performance with revenue of INR 847.3 crores, representing a 14.2% year-over-year growth. EBITDA margin improved to 22.8% from 20.1% in the previous quarter, driven by operational efficiencies and favorable currency movements.

1. Revenue Breakdown

1.1 By Service Line
- Digital Engineering: INR 338.9 crores (40% of revenue)
- Cloud & Infrastructure: INR 211.8 crores (25% of revenue)
- Data & Analytics: INR 169.5 crores (20% of revenue)
- Cybersecurity: INR 84.7 crores (10% of revenue)
- Consulting: INR 42.4 crores (5% of revenue)

1.2 By Geography
- North America: INR 508.4 crores (60%)
- Europe: INR 169.5 crores (20%)
- Asia Pacific: INR 127.1 crores (15%)
- Rest of World: INR 42.3 crores (5%)

1.3 By Client Segment
- Enterprise (Fortune 500): INR 423.7 crores (50%)
- Mid-Market: INR 254.2 crores (30%)
- SMB: INR 169.4 crores (20%)

2. Profitability

2.1 Key Margins
- Gross Margin: 38.4%
- EBITDA Margin: 22.8%
- Net Profit Margin: 16.3%
- Operating Cash Flow: INR 172.5 crores

2.2 Cost Structure
- Employee costs: 58.2% of revenue
- Infrastructure & technology: 12.4% of revenue
- Sales & marketing: 6.8% of revenue
- General & administrative: 3.4% of revenue
- Depreciation & amortization: 4.1% of revenue

3. Employee Metrics

3.1 Headcount
- Total employees: 12,847
- Net additions in Q3: 423
- Attrition rate (quarterly annualized): 13.8%
- Utilization rate: 82.4%

3.2 Compensation
- Average revenue per employee: INR 6.59 lakhs/month
- Median salary increase (last cycle): 11.2%

4. Client Metrics

4.1 Client Concentration
- Top client: 8.2% of revenue
- Top 5 clients: 28.4% of revenue
- Top 10 clients: 41.7% of revenue
- Total active clients: 287

4.2 New Business
- New client wins in Q3: 34
- Total Contract Value (TCV) of new deals: INR 312.5 crores
- Average deal size: INR 9.2 crores
- Win rate: 32.4%

5. Balance Sheet Highlights

5.1 Cash Position
- Cash and equivalents: INR 1,247.8 crores
- Short-term investments: INR 456.3 crores
- Total liquid assets: INR 1,704.1 crores
- Net debt: Zero (debt-free company)

5.2 Receivables
- Days Sales Outstanding (DSO): 67 days
- Unbilled revenue: INR 234.6 crores

6. Guidance for Q4 FY2024

- Revenue growth guidance: 3-5% QoQ
- EBITDA margin guidance: 22-24%
- Planned capex: INR 45 crores
- Expected headcount addition: 300-400

7. Risk Factors

- Currency volatility: 42% of revenue in USD, natural hedge covers 65%
- Client concentration risk in top 5 accounts
- Visa regulation changes impacting onsite delivery model
- Potential margin pressure from wage inflation in India
"""


# ── Test Framework ──────────────────────────────────────────────────

@dataclass
class TestCase:
    """Single RAG evaluation test case."""
    category: str
    question: str
    expected_keywords: list[str]  # Must appear in the answer
    forbidden_keywords: list[str] = field(default_factory=list)  # Must NOT appear
    should_refuse: bool = False   # Model should say "documents don't contain"

@dataclass
class TestResult:
    test: TestCase
    answer: str
    sources: list[dict]
    latency_seconds: float
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


TEST_CASES = [
    # ── Factual Retrieval (exact numbers from docs) ──
    TestCase(
        category="factual",
        question="How many days of annual leave are full-time employees entitled to?",
        expected_keywords=["24"],
    ),
    TestCase(
        category="factual",
        question="What is the p99 latency of Project Aurora?",
        expected_keywords=["50"],
    ),
    TestCase(
        category="factual",
        question="What was TechVista's Q3 revenue?",
        expected_keywords=["847"],
    ),
    TestCase(
        category="factual",
        question="How many Kafka brokers per region does Project Aurora use?",
        expected_keywords=["12"],
    ),
    TestCase(
        category="factual",
        question="What is the attrition rate mentioned in the financial report?",
        expected_keywords=["13.8"],
    ),

    # ── Multi-fact Reasoning (needs info from one section) ──
    TestCase(
        category="reasoning",
        question="What is the maximum performance bonus percentage and what rating do you need?",
        expected_keywords=["25", "5", "exceptional"],
    ),
    TestCase(
        category="reasoning",
        question="How much total liquid assets does TechVista have? Break it down.",
        expected_keywords=["1704", "1247", "456"],
    ),
    TestCase(
        category="reasoning",
        question="What encryption standards does Project Aurora use for data at rest and in transit?",
        expected_keywords=["aes-256", "tls 1.3"],
    ),

    # ── Cross-document Questions ──
    TestCase(
        category="cross-doc",
        question="Compare the hotel allowance for domestic metro travel with the international high-cost city allowance.",
        expected_keywords=["8000", "8,000", "300"],
    ),

    # ── Refusal Tests (should say docs don't contain info) ──
    TestCase(
        category="refusal",
        question="What is the capital of France?",
        expected_keywords=[],
        should_refuse=True,
    ),
    TestCase(
        category="refusal",
        question="Who is the CEO of ACME Corporation?",
        expected_keywords=[],
        should_refuse=True,
    ),
    TestCase(
        category="refusal",
        question="What programming language is Project Aurora built in?",
        expected_keywords=[],
        should_refuse=True,
    ),

    # ── Edge Cases ──
    TestCase(
        category="edge",
        question="What is the RPO and RTO for disaster recovery?",
        expected_keywords=["5 minute", "30 minute"],
    ),
    TestCase(
        category="edge",
        question="What percentage of TechVista's revenue comes from North America?",
        expected_keywords=["60"],
    ),
    TestCase(
        category="edge",
        question="How many WAF rules are active?",
        expected_keywords=["247"],
    ),
    TestCase(
        category="edge",
        question="What is the monthly internet reimbursement for remote workers?",
        expected_keywords=["1,500", "1500"],
    ),

    # ── Hallucination Trap (question is close to doc content but answer isn't there) ──
    TestCase(
        category="hallucination_trap",
        question="What is the employee stock option (ESOP) policy at ACME Corporation?",
        expected_keywords=[],
        should_refuse=True,
    ),
    TestCase(
        category="hallucination_trap",
        question="What database does Project Aurora use for its machine learning pipeline?",
        expected_keywords=[],
        should_refuse=True,
    ),
]


def register_user(client: httpx.Client) -> str:
    """Register and login, return JWT token."""
    username = f"stress_tester_{int(time.time())}"
    password = "TestPassword123!"
    
    # Register
    resp = client.post(f"{BASE_URL}/auth/register", json={
        "username": username, "password": password
    })
    if resp.status_code not in (200, 201, 409):
        print(f"Registration failed: {resp.status_code} {resp.text}")
        # Try login anyway
    
    # Login
    resp = client.post(f"{BASE_URL}/auth/login", json={
        "username": username, "password": password
    })
    if resp.status_code != 200:
        raise RuntimeError(f"Login failed: {resp.status_code} {resp.text}")
    
    return resp.json()["access_token"]


def upload_documents(client: httpx.Client, token: str) -> list[dict]:
    """Upload all test documents."""
    headers = {"Authorization": f"Bearer {token}"}
    
    docs = [
        ("company_policy.txt", DOC_COMPANY_POLICY),
        ("technical_spec.txt", DOC_TECHNICAL_SPEC),
        ("financial_report.txt", DOC_FINANCIAL_REPORT),
    ]
    
    files = [("files", (name, content.encode(), "text/plain")) for name, content in docs]
    
    print(f"  Uploading {len(docs)} documents...")
    resp = client.post(
        f"{BASE_URL}/documents/batch",
        headers=headers,
        files=files,
        timeout=120.0,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed: {resp.status_code} {resp.text}")
    
    result = resp.json()
    docs_list = result.get("documents", [])
    print(f"  Uploaded {len(docs_list)} documents successfully")
    for doc in docs_list:
        print(f"    - {doc['filename']}: {doc['chunk_count']} chunks, {doc.get('page_count', '?')} pages")
    return docs_list


def query_rag(client: httpx.Client, token: str, question: str) -> tuple[str, list[dict], float]:
    """Send a RAG query and collect the streamed response."""
    headers = {"Authorization": f"Bearer {token}"}
    
    start = time.time()
    resp = client.post(
        f"{BASE_URL}/rag/query/stream",
        headers=headers,
        json={"question": question},
        timeout=TIMEOUT,
    )
    
    answer_parts = []
    sources = []
    
    for line in resp.text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        if event.get("type") == "token":
            answer_parts.append(event.get("content", ""))
        elif event.get("type") == "sources":
            sources = event.get("sources", [])
        elif event.get("type") == "error":
            answer_parts.append(f"[ERROR: {event.get('message', 'unknown')}]")
    
    elapsed = time.time() - start
    answer = "".join(answer_parts).strip()
    return answer, sources, elapsed


def evaluate_answer(test: TestCase, answer: str, sources: list[dict], latency: float) -> TestResult:
    """Evaluate a RAG answer against expected criteria."""
    import re
    failures = []
    
    def normalize(text: str) -> str:
        # Lowercase, and strip commas between digits to make numeric checks robust
        t = text.lower()
        t = re.sub(r'(\d),(\d)', r'\1\2', t)
        return t

    answer_norm = normalize(answer)
    
    # Check refusal
    if test.should_refuse:
        refusal_phrases = [
            "do not contain", "does not contain", "don't contain",
            "do not provide", "does not provide", "don't provide",
            "not enough information", "no relevant", "could not find",
            "not contain information", "not available in",
            "no information", "not mentioned",
        ]
        refused = any(phrase in answer_norm for phrase in refusal_phrases)
        if not refused:
            failures.append(f"HALLUCINATION: Should have refused but answered: {answer[:200]}")
    else:
        # Check expected keywords
        normalized_expected = [normalize(kw) for kw in test.expected_keywords]
        unique_expected = list(dict.fromkeys(normalized_expected))
        for kw in unique_expected:
            if kw not in answer_norm:
                failures.append(f"MISSING_KEYWORD: '{kw}' not found in answer")
        
        # Check forbidden keywords
        for kw in test.forbidden_keywords:
            if normalize(kw) in answer_norm:
                failures.append(f"FORBIDDEN_KEYWORD: '{kw}' found in answer")
        
        # Check sources
        if not sources:
            failures.append("NO_SOURCES: No source chunks returned")
    
    # Latency check
    if latency > 60:
        failures.append(f"SLOW: Response took {latency:.1f}s (>60s)")
    
    return TestResult(
        test=test,
        answer=answer,
        sources=sources,
        latency_seconds=latency,
        passed=len(failures) == 0,
        failure_reasons=failures,
    )


def run_stress_test():
    """Main test runner."""
    print("=" * 70)
    print("RAG PIPELINE STRESS TEST")
    print("=" * 70)
    
    with httpx.Client(timeout=TIMEOUT) as client:
        # 1. Health check
        print("\n[1/4] Health check...")
        try:
            resp = client.get(f"{BASE_URL}/health", timeout=10)
            print(f"  Server status: {resp.status_code}")
        except Exception as e:
            print(f"  FATAL: Server not reachable: {e}")
            return
        
        # 2. Register user
        print("\n[2/4] Registering test user...")
        token = register_user(client)
        print(f"  Token obtained: {token[:20]}...")
        
        # 3. Upload documents
        print("\n[3/4] Uploading test documents...")
        try:
            docs = upload_documents(client, token)
        except Exception as e:
            print(f"  FATAL: Upload failed: {e}")
            return
        
        # Wait for indexing
        print("  Waiting 3s for index build...")
        time.sleep(3)
        
        # 4. Run test queries
        print(f"\n[4/4] Running {len(TEST_CASES)} test queries...")
        print("-" * 70)
        
        results: list[TestResult] = []
        for i, test in enumerate(TEST_CASES, 1):
            print(f"\n  [{i}/{len(TEST_CASES)}] [{test.category}] {test.question}")
            try:
                answer, sources, latency = query_rag(client, token, test.question)
                result = evaluate_answer(test, answer, sources, latency)
                results.append(result)
                
                status = "PASS" if result.passed else "FAIL"
                print(f"  [{status}] ({latency:.1f}s) sources={len(sources)}")
                if not result.passed:
                    for reason in result.failure_reasons:
                        print(f"    - {reason}")
                print(f"  Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            except Exception as e:
                print(f"  [ERROR]: {e}")
                results.append(TestResult(
                    test=test, answer=f"ERROR: {e}", sources=[], 
                    latency_seconds=0, passed=False,
                    failure_reasons=[f"EXCEPTION: {e}"]
                ))
    
    # ── Print Summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    categories: dict[str, list[TestResult]] = {}
    for r in results:
        cat = r.test.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    print(f"\nOverall: {passed}/{total} passed ({100*passed/total:.0f}%)")
    print(f"Average latency: {sum(r.latency_seconds for r in results)/total:.1f}s")
    
    for cat, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r.passed)
        print(f"  {cat}: {cat_passed}/{len(cat_results)}")
    
    if failed > 0:
        print(f"\n--- FAILURES ({failed}) ---")
        for r in results:
            if not r.passed:
                print(f"\n  Q: {r.test.question}")
                print(f"  Category: {r.test.category}")
                for reason in r.failure_reasons:
                    print(f"  - {reason}")
                print(f"  Answer (first 300 chars): {r.answer[:300]}")
    
    # Write full results to file
    output = {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{100*passed/total:.1f}%",
            "avg_latency": f"{sum(r.latency_seconds for r in results)/total:.1f}s",
        },
        "by_category": {
            cat: {
                "passed": sum(1 for r in cat_results if r.passed),
                "total": len(cat_results),
            }
            for cat, cat_results in categories.items()
        },
        "details": [
            {
                "category": r.test.category,
                "question": r.test.question,
                "passed": r.passed,
                "latency": f"{r.latency_seconds:.1f}s",
                "failure_reasons": r.failure_reasons,
                "answer": r.answer,
                "source_count": len(r.sources),
                "source_scores": [f"{s.get('score', 0):.4f}" for s in r.sources],
            }
            for r in results
        ]
    }
    
    out_path = Path(__file__).parent / "stress_test_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    run_stress_test()
