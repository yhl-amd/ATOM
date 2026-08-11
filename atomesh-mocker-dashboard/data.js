window.BENCHMARK_DATA = {
  "lastUpdate": 1786478665382,
  "repoUrl": "https://github.com/yhl-amd/ATOM",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "chenxjhit@163.com",
            "name": "Phi-C",
            "username": "Phi-C"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T15:22:45+08:00",
          "tree_id": "09dee0909acc7df1cffcd23f3d5e09203bfdb1b8",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1785324083130,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2316.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417008 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417008 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417008 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417008 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417008 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 8724.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570415 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570415 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.33,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570415 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.24,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570415 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570415 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3820.31,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687655 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687655 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687655 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687655 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687655 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5789.92,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042185 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042185 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042185 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042185 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042185 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7380.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.91,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.42,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2292.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 8535.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1536431 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.83,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1536431 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1536431 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.3,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1536431 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1536431 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3808.72,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=685569 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=685569 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=685569 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=685569 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=685569 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5815.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046716 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046716 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046716 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.38,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046716 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046716 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 7444.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1340088 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1340088 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.89,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1340088 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.39,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1340088 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1340088 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2304.99,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414899 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414899 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414899 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414899 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414899 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 8639.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1555192 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.81,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1555192 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1555192 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.27,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1555192 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1555192 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3816.43,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=686957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=686957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=686957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=686957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=686957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5749.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1034831 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1034831 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1034831 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1034831 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1034831 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 7494.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1349047 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1349047 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1349047 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.37,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1349047 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1349047 Run: https://github.com/yhl-amd/ATOM/actions/runs/30443401713"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1785356141955,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2191.23,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=394421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=394421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=394421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=394421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=394421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7310.03,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315805 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315805 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 4.01,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315805 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.09,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315805 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315805 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3563.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=641351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=641351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=641351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=641351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=641351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5289.43,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6627.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193034 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193034 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193034 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193034 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193034 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2152.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387506 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387506 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387506 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387506 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387506 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7422.3,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.01,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3523.64,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634255 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634255 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634255 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634255 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634255 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5297.8,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953604 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953604 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953604 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953604 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953604 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6684.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1203209 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1203209 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.13,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1203209 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.69,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1203209 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1203209 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2134.15,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384147 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384147 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384147 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384147 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384147 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7411.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334078 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334078 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334078 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 5.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334078 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1334078 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3514.74,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632653 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632653 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632653 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.86,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632653 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632653 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5213.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938438 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938438 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938438 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938438 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938438 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6600.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188112 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188112 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188112 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188112 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188112 Run: https://github.com/yhl-amd/ATOM/actions/runs/30483758956"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1785443107185,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2135.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384320 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384320 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384320 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384320 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384320 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7460.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342897 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342897 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.92,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342897 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.01,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342897 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342897 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3473.26,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625187 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625187 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625187 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625187 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625187 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5162.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929317 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929317 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929317 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929317 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929317 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6527.85,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1175013 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1175013 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1175013 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1175013 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1175013 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2152.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387523 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387523 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387523 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387523 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387523 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7563.21,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361377 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.07,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361377 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.86,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361377 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361377 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361377 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3471.31,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624835 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624835 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624835 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624835 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624835 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5223.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=940219 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=940219 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=940219 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=940219 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=940219 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6470.05,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1164609 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1164609 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1164609 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.78,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1164609 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1164609 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2160.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7529.79,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1355362 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1355362 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1355362 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.95,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1355362 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1355362 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3494.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629016 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629016 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629016 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.86,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629016 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629016 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5242.2,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943596 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943596 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943596 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943596 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943596 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6583.01,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184941 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184941 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184941 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184941 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184941 Run: https://github.com/yhl-amd/ATOM/actions/runs/30574861135"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1785529487255,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2144.8,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386064 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386064 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386064 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386064 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386064 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7538.51,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356932 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356932 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.86,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356932 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.92,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356932 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356932 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3482.18,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626792 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626792 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626792 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626792 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626792 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5166.02,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929883 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929883 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.24,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929883 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929883 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929883 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6634.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194186 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194186 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194186 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.71,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194186 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194186 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2135.67,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384421 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7518.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1353351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1353351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.88,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1353351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.94,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1353351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1353351 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3449.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620956 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620956 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620956 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620956 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620956 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5168.56,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930340 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930340 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930340 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930340 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930340 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6620.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1191727 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1191727 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1191727 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1191727 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1191727 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2133.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383957 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7502.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350458 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350458 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350458 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350458 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350458 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3465.04,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623707 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623707 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623707 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623707 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623707 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5170.72,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930730 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930730 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930730 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930730 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930730 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6636.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194541 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194541 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194541 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194541 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194541 Run: https://github.com/yhl-amd/ATOM/actions/runs/30659093794"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1785614827876,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2145.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386111 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386111 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386111 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386111 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386111 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7538.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1357014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.07,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1357014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1357014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.96,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1357014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1357014 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3499.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629839 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629839 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629839 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629839 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629839 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5194.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935068 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935068 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.24,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935068 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935068 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935068 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6500.59,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170107 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170107 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170107 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170107 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170107 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2116.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381036 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381036 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381036 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381036 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381036 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7451.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1341264 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.1,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1341264 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1341264 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.02,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1341264 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1341264 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3453.74,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=621674 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=621674 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=621674 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=621674 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=621674 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5205.52,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936993 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936993 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936993 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936993 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936993 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6538.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1176936 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1176936 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.17,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1176936 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1176936 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1176936 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2122.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382098 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7536.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356611 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356611 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356611 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.96,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356611 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1356611 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3461.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=622991 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=622991 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=622991 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=622991 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=622991 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5302.39,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=954430 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=954430 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=954430 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=954430 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=954430 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6631.74,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193713 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193713 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193713 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.8,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193713 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193713 Run: https://github.com/yhl-amd/ATOM/actions/runs/30714135598"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1785701166216,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 4634.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=834205 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.2,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=834205 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.26,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=834205 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=834205 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=834205 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 16445.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2960237 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 0.94,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2960237 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 1.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2960237 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 2.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2960237 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2960237 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 7537.36,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1356724 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.25,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1356724 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1356724 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1356724 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1356724 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 11506.99,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2071258 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.33,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2071258 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2071258 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 0.91,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2071258 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2071258 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 14108.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2539462 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2539462 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.07,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2539462 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2539462 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2539462 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 4569.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=822584 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.21,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=822584 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.26,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=822584 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.38,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=822584 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=822584 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 16817.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=3027171 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 0.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=3027171 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 1.85,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=3027171 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 2.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=3027171 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=3027171 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 7328.38,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1319109 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.26,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1319109 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1319109 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1319109 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1319109 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 11365.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2045752 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.33,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2045752 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2045752 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2045752 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2045752 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 14810.21,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2665838 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2665838 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.02,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2665838 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2665838 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2665838 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 4693.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=844757 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.2,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=844757 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.25,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=844757 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=844757 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=844757 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 15955.43,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2871977 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 0.97,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2871977 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 1.97,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2871977 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 2.67,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2871977 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2871977 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 7475.92,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1345665 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.25,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1345665 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1345665 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1345665 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1345665 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 11212.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2018327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2018327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2018327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 0.99,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2018327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2018327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 13659.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2458767 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2458767 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.12,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2458767 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.69,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2458767 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2458767 Run: https://github.com/yhl-amd/ATOM/actions/runs/30762779373"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1785789411150,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2158.7,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388566 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388566 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388566 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388566 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388566 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7476.84,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1345832 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.09,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1345832 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1345832 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1345832 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1345832 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3506.86,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631234 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631234 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631234 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631234 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631234 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5285.4,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=951372 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=951372 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=951372 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=951372 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=951372 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6662.61,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1199269 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1199269 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.13,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1199269 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.71,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1199269 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1199269 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2146.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386392 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386392 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386392 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386392 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386392 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7664.67,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379641 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.04,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379641 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.8,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379641 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.84,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379641 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379641 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3502.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630496 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630496 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630496 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630496 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630496 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5341.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961544 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961544 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961544 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961544 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961544 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6715.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208827 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208827 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208827 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.69,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208827 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208827 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2144.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386043 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386043 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386043 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386043 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=386043 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7491.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1348502 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.09,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1348502 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1348502 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.93,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1348502 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1348502 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3489.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=628031 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=628031 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=628031 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=628031 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=628031 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5194.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935060 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935060 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935060 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935060 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935060 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6638.18,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194873 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194873 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194873 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.7,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194873 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1194873 Run: https://github.com/yhl-amd/ATOM/actions/runs/30846647342"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1785875839495,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2734.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=492164 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=492164 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=492164 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=492164 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=492164 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 10882.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1958923 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1958923 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.68,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1958923 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.37,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1958923 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1958923 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 4651.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=837356 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=837356 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=837356 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=837356 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=837356 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 7163.33,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1289399 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1289399 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1289399 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1289399 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1289399 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 9246.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1664393 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1664393 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1664393 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.96,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1664393 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1664393 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2697.64,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=485576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=485576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=485576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=485576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=485576 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 10892.61,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1960669 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.42,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1960669 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.71,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1960669 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.42,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1960669 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1960669 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 4664.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=839618 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=839618 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=839618 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=839618 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=839618 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 6948.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1250702 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1250702 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.89,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1250702 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1250702 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1250702 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 9267.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1668143 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1668143 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1668143 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1668143 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1668143 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2674.04,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=481327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=481327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=481327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=481327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=481327 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 10870.6,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1956708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1956708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.69,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1956708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.37,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1956708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1956708 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 4592.51,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=826652 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=826652 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=826652 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=826652 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=826652 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 7134.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1284242 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1284242 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1284242 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1284242 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1284242 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 9203.03,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1656545 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1656545 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1656545 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1656545 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1656545 Run: https://github.com/yhl-amd/ATOM/actions/runs/30944199729"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1785962055979,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2264.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407541 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407541 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407541 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407541 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407541 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 8697.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1565557 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1565557 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.34,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1565557 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.26,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1565557 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1565557 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3752.04,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675367 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675367 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675367 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.84,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675367 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675367 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5684.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1023195 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1023195 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1023195 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1023195 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1023195 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7288.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1311914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1311914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1311914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1311914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1311914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2239.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=403166 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=403166 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=403166 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=403166 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=403166 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 8517.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1533118 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.83,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1533118 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.42,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1533118 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.35,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1533118 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1533118 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3740.23,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=673242 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=673242 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.7,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=673242 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.81,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=673242 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=673242 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5668.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1020328 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1020328 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1020328 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1020328 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1020328 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 7427.83,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1337009 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1337009 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.9,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1337009 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1337009 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1337009 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2238.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=402928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=402928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=402928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=402928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=402928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 8516.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1532976 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.83,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1532976 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1532976 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1532976 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1532976 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3727.01,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=670862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=670862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=670862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.82,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=670862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=670862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5701.74,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1026313 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1026313 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1026313 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1026313 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1026313 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 7381.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328624 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328624 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.91,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328624 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.4,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328624 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1328624 Run: https://github.com/yhl-amd/ATOM/actions/runs/31040133557"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1786064517817,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2260.1,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406818 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406818 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406818 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406818 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406818 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7523.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354228 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354228 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.88,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354228 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.91,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354228 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354228 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3663.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=659496 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=659496 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=659496 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=659496 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=659496 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5392.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970658 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970658 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970658 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970658 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970658 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6728.66,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211159 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211159 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211159 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.67,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211159 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211159 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2275.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409674 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409674 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409674 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409674 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409674 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7482.88,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.09,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.89,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.94,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3625.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=652551 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=652551 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=652551 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.84,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=652551 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=652551 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5359.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=964696 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=964696 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=964696 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=964696 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=964696 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6691.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1204507 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1204507 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1204507 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.7,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1204507 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1204507 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2261.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407157 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407157 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407157 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407157 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407157 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7547.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1358633 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.07,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1358633 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1358633 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.92,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1358633 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1358633 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3616.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=650984 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=650984 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=650984 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.84,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=650984 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=650984 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5393.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=970862 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6730.7,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211526 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211526 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211526 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211526 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1211526 Run: https://github.com/yhl-amd/ATOM/actions/runs/31133164623"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1786132623601,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2955.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=532069 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.32,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=532069 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=532069 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=532069 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=532069 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 11186.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2013490 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2013490 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2013490 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.34,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2013490 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2013490 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 4912.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884308 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884308 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884308 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884308 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884308 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 7465.5,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1343790 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1343790 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1343790 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1343790 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1343790 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 9638.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734891 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734891 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734891 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.88,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734891 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734891 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2949.52,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=530914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.32,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=530914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=530914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=530914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=530914 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 11345.5,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2042190 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.38,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2042190 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.57,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2042190 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.31,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2042190 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2042190 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 4919.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=885543 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=885543 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=885543 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=885543 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=885543 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 7455.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1342056 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1342056 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1342056 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1342056 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1342056 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 9541.48,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1717467 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.81,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1717467 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1717467 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.89,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1717467 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1717467 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2938.43,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=528918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.33,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=528918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=528918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=528918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=528918 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 11286.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2031594 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.38,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2031594 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2031594 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.32,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2031594 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2031594 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 4916.51,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884971 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884971 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884971 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884971 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=884971 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 7475.56,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1345601 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1345601 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.84,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1345601 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1345601 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1345601 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 9535.74,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1716433 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.81,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1716433 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1716433 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1716433 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1716433 Run: https://github.com/yhl-amd/ATOM/actions/runs/31209362438"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1786217939204,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2082.83,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374910 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374910 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374910 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374910 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374910 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7283.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1311052 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1311052 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1311052 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.06,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1311052 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1311052 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3392.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=610618 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=610618 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=610618 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=610618 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=610618 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5107.33,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919320 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919320 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.24,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919320 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919320 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919320 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6524.07,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174332 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174332 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.17,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174332 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174332 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174332 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2080.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374484 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374484 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374484 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374484 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374484 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7407.75,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333395 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333395 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333395 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.97,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333395 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333395 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3409.23,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613662 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613662 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613662 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613662 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613662 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5150.22,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927039 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927039 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927039 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927039 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927039 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6450.2,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161036 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161036 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161036 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161036 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161036 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2108.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379468 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379468 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379468 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379468 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379468 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7511.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1352128 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1352128 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1352128 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.92,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1352128 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1352128 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3425.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616623 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616623 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616623 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616623 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616623 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5123.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=922197 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=922197 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.24,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=922197 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=922197 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=922197 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6505.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171063 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171063 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171063 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171063 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171063 Run: https://github.com/yhl-amd/ATOM/actions/runs/31272399790"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1786304567151,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2156.79,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388222 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388222 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388222 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388222 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388222 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7505.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.9,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.97,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3507.85,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631413 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631413 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631413 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 1.02,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631413 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=631413 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5263.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=947462 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=947462 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=947462 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=947462 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=947462 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6660.57,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198902 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198902 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198902 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.8,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198902 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198902 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2178.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392187 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392187 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392187 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392187 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392187 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7350.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1323095 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1323095 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.98,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1323095 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.05,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1323095 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1323095 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3477.99,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626038 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626038 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626038 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626038 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=626038 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5239.8,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943164 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943164 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943164 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943164 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=943164 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6590.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1186338 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1186338 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1186338 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1186338 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1186338 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2143.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385797 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385797 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385797 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385797 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385797 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7666.27,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.04,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.84,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1379928 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3496.84,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629432 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629432 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629432 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.86,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629432 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629432 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5248.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=944778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=944778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=944778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=944778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=944778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6627.15,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1192887 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1192887 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1192887 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1192887 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1192887 Run: https://github.com/yhl-amd/ATOM/actions/runs/31329657891"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1786391941355,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2112.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=380274 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=380274 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=380274 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=380274 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=380274 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7388.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1329861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1329861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1329861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.01,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1329861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1329861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3442.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619648 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619648 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619648 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619648 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619648 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5196.38,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935349 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935349 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935349 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935349 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935349 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6510.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171861 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2128.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383145 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383145 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383145 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383145 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383145 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7484.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1347170 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.09,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1347170 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.89,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1347170 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.98,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1347170 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1347170 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3476.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625756 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625756 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625756 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625756 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=625756 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5213.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938517 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938517 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938517 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938517 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938517 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6630.05,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193409 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193409 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193409 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193409 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193409 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2123.95,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382311 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382311 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382311 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382311 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382311 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7306.88,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315239 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315239 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.99,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315239 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 5.06,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315239 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315239 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3448.33,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5195.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935143 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935143 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935143 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935143 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935143 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6495.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1169130 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1169130 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1169130 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1169130 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1169130 Run: https://github.com/yhl-amd/ATOM/actions/runs/31421731296"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18",
          "message": "[CI][Doc] add ci case and recipe for glm5.2 agentic benchmark (#1675)\n\n* [CI][Doc] add ci case and recipe for glm5.2 agentic benchmark\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* use prefix cache as default setting\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* update\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* 1. disable lmcache; 2. add 1P2D con=4, TP4 case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-29T07:22:45Z",
          "url": "https://github.com/yhl-amd/ATOM/commit/4e8c1021e6615d1f8a6c5b1a0e1967cd56007b18"
        },
        "date": 1786478664185,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2154.27,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387768 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387768 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387768 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387768 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=387768 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7410.45,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333881 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333881 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.99,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333881 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.12,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333881 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333881 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3544.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638024 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638024 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638024 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638024 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638024 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5339.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961040 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961040 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961040 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961040 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=961040 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6659.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.13,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.69,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2179.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392333 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392333 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392333 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392333 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=392333 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7457.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342405 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.1,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342405 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342405 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.03,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342405 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342405 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3503.89,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630700 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5004.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=900778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=900778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=900778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 2.89,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=900778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=900778 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6525.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174523 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174523 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.25,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174523 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.97,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174523 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174523 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 1927.07,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=346873 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=346873 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=346873 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 3.28,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=346873 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=346873 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 6213.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1118429 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1118429 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 5.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1118429 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 7.59,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1118429 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1118429 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 2768.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=498336 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=498336 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=498336 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 4.24,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=498336 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=498336 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 4767.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=858110 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.81,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=858110 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=858110 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 3.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=858110 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=858110 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 5177.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=931904 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=931904 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 3.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=931904 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 5.96,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=931904 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=931904 Run: https://github.com/yhl-amd/ATOM/actions/runs/31525850653"
          }
        ]
      }
    ]
  }
}