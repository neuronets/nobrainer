#!bash

set -xe

GH_TOKEN=$(cat gh_token)

sudo ./svc.sh stop
sudo ./svc.sh uninstall

remove_token=$(curl -L \
                    -X POST \
                    -H "Accept: application/vnd.github+json" \
                    -H "Authorization: Bearer ${GH_TOKEN}" \
                    -H "X-GitHub-Api-Version: 2022-11-28" \
                    https://api.github.com/orgs/neuronets/actions/runners/remove-token \
                   | jq -r ".token")

./config.sh remove --token ${remove_token}
