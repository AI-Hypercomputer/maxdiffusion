# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Clean up Python codes using Pylint & Pyink
# Googlers: please run `sudo apt install pipx; pipx install pylint --force; pipx install pyink==23.10.0` in advance

set -e # Exit immediately if any command fails

FOLDERS_TO_FORMAT=("src/maxdiffusion" "end_to_end/tpu")
LINE_LENGTH=$(grep -E "^max-line-length=" pylintrc | cut -d '=' -f 2)
# Keep in sync with .pre-commit-config.yaml and .github/workflows/CPUTests.yml.
EXPECTED_PYINK_VERSION="23.10.0"

# Different pyink versions can format differently; warn if the local version won't match CI.
INSTALLED_PYINK_VERSION=$(pyink --version 2>/dev/null | head -n 1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
if [[ "${INSTALLED_PYINK_VERSION}" != "${EXPECTED_PYINK_VERSION}" ]]; then
  echo -e "\e[33mWARNING: CI formats with pyink ${EXPECTED_PYINK_VERSION} but you have '${INSTALLED_PYINK_VERSION:-none}'." \
          "Results may differ from CI. Install the pinned version with: pip install pyink==${EXPECTED_PYINK_VERSION}\e[0m"
fi

# Check for --check flag
CHECK_ONLY_PYINK_FLAGS=""
if [[ "$1" == "--check" ]]; then
  CHECK_ONLY_PYINK_FLAGS="--check --diff --color"
fi

for folder in "${FOLDERS_TO_FORMAT[@]}"
do
  pyink "$folder" ${CHECK_ONLY_PYINK_FLAGS} --pyink-indentation=2 --line-length=${LINE_LENGTH}
done

for folder in "${FOLDERS_TO_FORMAT[@]}"
do
  # pylint doesn't change files, only reports errors.
  pylint "./$folder"
done

echo "Successfully clean up all codes."
