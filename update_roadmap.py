#!/usr/bin/env python3
"""Update roadmap.html with E8 Sensor Panel v2 results."""

with open("roadmap.html", "r") as f:
    html = f.read()

# 1. Update Skeletal Core metric bar - add v2 result
old_metric = """                    <span class="label">Geometric substrate:</span>
                    <span class="value">85% pretest &rarr; 100% Branch Quality (reference baseline)</span>"""
new_metric = """                    <span class="label">Geometric substrate (v1):</span>
                    <span class="value">85% pretest (trunk classification) &rarr; 100% Branch Quality</span><br>
                    <span class="label">E8 Sensor Panel (v2):</span>
                    <span class="value">98.6% pretest (exact grid match, 3214/3260 pairs, 1009 tasks)</span>"""
html = html.replace(old_metric, new_metric)

# 2. Mark sensor panel architecture design as done in Harmonic Stack V2
html = html.replace(
    '<li>Sensor panel architecture design</li>',
    '<li class="done">Sensor panel architecture design &mdash; dual-panel pathway validated at 98.6%</li>'
)

# 3. Add v2 measurement to Methodology
old_meth = '<li class="done">85% pretest accuracy measured on ARC evaluation set (not theoretical projection)</li>'
new_meth = old_meth + '\n                    <li class="done">98.6% pretest accuracy via E8 Sensor Panel v2 &mdash; exact grid match on 1009 ARC tasks (3214/3260 pairs)</li>'
html = html.replace(old_meth, new_meth)

# 4. Update timestamp
html = html.replace('Last updated: 2026-02-06', 'Last updated: 2026-02-06 14:00 EST')

with open("roadmap.html", "w") as f:
    f.write(html)

print("EDITS APPLIED")
print(f"File size: {len(html)} bytes")
