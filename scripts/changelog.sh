#!/bin/sh
#
# Print the CHANGELOG.md section for a version.
#
#   scripts/changelog.sh v0.2.0
#
# Exits non-zero when there is no such section, which is how the release
# workflow refuses to cut a tag nobody wrote a line for. Runnable by hand for
# the same reason: the answer CI acts on should be one you can see first.
set -eu

[ $# -eq 1 ] || { echo "usage: $0 <version>" >&2; exit 2; }

# Accept v0.2.0 and 0.2.0 alike; the heading is written without the v.
version="${1#v}"
file="${CHANGELOG:-CHANGELOG.md}"

[ -f "$file" ] || { echo "no $file" >&2; exit 1; }

body="$(
  awk -v want="## [$version]" '
    index($0, want) == 1 { found = 1; next }
    # The next release heading ends this section, and so does the block of
    # link definitions at the foot of the file -- without that second rule the
    # oldest release would carry every link in the file into its release notes.
    found && (/^## / || /^\[[^]]+\]:/) { exit }
    found { print }
  ' "$file"
)"

# Strip leading and trailing blank lines, then check there is anything left: a
# heading with nothing under it is not a described release.
body="$(printf '%s\n' "$body" | sed -e '/./,$!d' | awk '{ lines[NR] = $0 }
  END { last = NR; while (last > 0 && lines[last] ~ /^[[:space:]]*$/) last--;
        for (i = 1; i <= last; i++) print lines[i] }')"

if [ -z "$body" ]; then
  echo "$file has no section for $version -- add '## [$version]' before tagging" >&2
  exit 1
fi

printf '%s\n' "$body"
