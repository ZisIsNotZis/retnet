#!/bin/bash
(
sudo find / -type f -iregex '.*\.\(py[ix]?\|f2py_f2cmap\)' -print0|sudo xargs -0 cat
sudo find / -type f -iname '*.ipynb' -print0|sudo xargs -0 jq -r .cells[].source[]
)|tr '[:upper:]\t' '[:lower:] '|tr -Cd ' abcdefghijklmnopqrstuvwxyz0123456789_."&|^+\\\-*/%@=()[]{}<>:\n#,~;!$'\'>py.txt &

sudo find / -type f -iregex '.*\.\(c\(c\|xx\|pp\)?\|h\(pp\)?\)' -print0|
sudo xargs -0 cat|tr '[:upper:]\t' '[:lower:] '|tr -Cd ' abcdefghijklmnopqrstuvwxyz0123456789_."&|^+\\\-*/%@=()[]{}<>:\n#,~;!$'\'>c.txt &

sudo find / -type f -iregex '.*\.\([cz]\|ba\|fi\)?sh' -print0|
sudo xargs -0 cat|tr '[:upper:]\t' '[:lower:] '|tr -Cd ' abcdefghijklmnopqrstuvwxyz0123456789_."&|^+\\\-*/%@=()[]{}<>:\n#,~;!$'\'>sh.txt &

sudo find / -type f -iname '*.java' -print0|
sudo xargs -0 cat|tr '[:upper:]\t' '[:lower:] '|tr -Cd ' abcdefghijklmnopqrstuvwxyz0123456789_."&|^+\\\-*/%@=()[]{}<>:\n#,~;!$'\'>java.txt &

sudo find / -type f -iregex '.*\.[jt]s' -print0|
sudo xargs -0 cat|tr '[:upper:]\t' '[:lower:] '|tr -Cd ' abcdefghijklmnopqrstuvwxyz0123456789_."&|^+\\\-*/%@=()[]{}<>:\n#,~;!$'\'>js.txt &

sudo find / -type f -iregex '.*\.p\(erl\|[lm]\)' -print0|
sudo xargs -0 cat|tr '[:upper:]\t' '[:lower:] '|tr -Cd ' abcdefghijklmnopqrstuvwxyz0123456789_."&|^+\\\-*/%@=()[]{}<>:\n#,~;!$'\'>perl.txt &

wait
