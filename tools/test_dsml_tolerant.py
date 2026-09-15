#!/usr/bin/env python3
"""The tolerant DSML parser keeps an HTML value whole and every parameter of the call."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from server.dsml_tolerant import parse_tolerant  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    fails += 0 if ok else 1


B = "｜DSML｜"
page = "<!DOCTYPE html>\n<html><head><style>body{margin:0}</style></head><body><canvas id=\"c\"></canvas>\n<script>const a=1<2;</script></body></html>"
text = (f"<{B} calls>\n<{B} invoke name=\"write\">\n"
        f"<{B} parameter name=\"filePath\" string=\"true\">/tmp/aquarium.html</{B} parameter>\n"
        f"<{B} parameter name=\"content\" string=\"true\">{page}</{B} parameter>\n"
        f"</{B} invoke>\n</{B} calls>")
calls = parse_tolerant(text)
check("one call is recovered", len(calls) == 1, str(calls)[:200])
args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
check("the path survives", args.get("filePath") == "/tmp/aquarium.html", str(args.keys()))
check("an HTML value runs to its closing tag, not to the first '<'", args.get("content") == page,
      repr(args.get("content"))[:120])
check("the tool name is kept", calls and calls[0]["function"]["name"] == "write")

# content first, path after: the order the checkpoint sometimes chooses
text2 = (f"<{B} invoke name=\"write\">\n"
         f"<{B} parameter name=\"content\" string=\"true\">{page}</{B} parameter>\n"
         f"<{B} parameter name=\"filePath\" string=\"true\">/tmp/b.html</{B} parameter>\n</{B} invoke>")
a2 = json.loads(parse_tolerant(text2)[0]["function"]["arguments"])
check("path after content is still found", a2.get("filePath") == "/tmp/b.html" and a2.get("content") == page)

# cut off before the closing tag: the value is what was written, the path is not invented
text3 = (f"<{B} invoke name=\"write\">\n<{B} parameter name=\"filePath\" string=\"true\">/tmp/c.html</{B} parameter>\n"
         f"<{B} parameter name=\"content\" string=\"true\"><html><body>half a page")
a3 = json.loads(parse_tolerant(text3)[0]["function"]["arguments"])
check("a value cut before its closing tag keeps what was written", a3.get("content") == "<html><body>half a page"
      and a3.get("filePath") == "/tmp/c.html")

# a non-string value
text4 = f"<{B} invoke name=\"search\">\n<{B} parameter name=\"limit\" string=\"false\">10</{B} parameter>\n</{B} invoke>"
check("a non-string value is decoded", json.loads(parse_tolerant(text4)[0]["function"]["arguments"]) == {"limit": 10})
print("all ok" if not fails else f"{fails} failed")
sys.exit(1 if fails else 0)
