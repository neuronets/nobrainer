#!bash

set -xe

GH_TOKEN=$(cat gh_token)

ar_url=$(curl https://github.com/actions/runner/releases | \
             grep -m 1 actions-runner-linux-x64 | \
             cut -d ' ' -f 4)
curl -O -L ${ar_url}
tar xzf $(basename ${ar_url})

reg_token=$(curl -L \
                 -X POST \
                 -H "Accept: application/vnd.github+json" \
                 -H "Authorization: Bearer ${GH_TOKEN}" \
                 -H "X-GitHub-Api-Version: 2022-11-28" \
                 https://api.github.com/orgs/neuronets/actions/runners/registration-token \
                | jq -r ".token")

runner_id=$(ec2metadata --instance-id)
./config.sh \
    --url https://github.com/neuronets \
    --token ${reg_token} \
    --name ${runner_id} \
    --labels ${runner_id} \
    --unattended

sudo ./svc.sh install
sudo ./svc.sh start
