#!/usr/bin/env bash
set -euo pipefail

tag="${1:-}"
requested_channel="${2:-}"

if [ -z "$tag" ]; then
  echo "release tag is required" >&2
  exit 1
fi

metadata="$(cargo metadata --locked --no-deps --format-version 1)"
crate_name="$(jq -r '.packages[0].name' <<< "$metadata")"
crate_version="$(jq -r '.packages[0].version' <<< "$metadata")"
expected_tag="v${crate_version}"

if [ "$tag" != "$expected_tag" ]; then
  echo "release tag $tag does not match Cargo.toml version $crate_version; expected $expected_tag" >&2
  exit 1
fi

if [[ "$crate_version" == *-* ]]; then
  channel="canary"
  prerelease="true"
  make_latest="false"
else
  channel="stable"
  prerelease="false"
  make_latest="true"
fi

if [ -n "$requested_channel" ] && [ "$requested_channel" != "$channel" ]; then
  echo "requested channel $requested_channel does not match crate version $crate_version ($channel)" >&2
  exit 1
fi

echo "tag=$tag"
echo "channel=$channel"
echo "prerelease=$prerelease"
echo "make_latest=$make_latest"
echo "crate_name=$crate_name"
echo "crate_version=$crate_version"
echo "crate_file=${crate_name}-${crate_version}.crate"
