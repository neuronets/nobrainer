# EC2 GPU Runner Setup

This document describes how to configure the AWS EC2 instance used as a
self-hosted GitHub Actions runner for GPU integration tests.

The workflow (`guide-notebooks-ec2.yml`) uses
[machulav/ec2-github-runner](https://github.com/machulav/ec2-github-runner) to
start an ephemeral EC2 instance, run GPU tests, and terminate the instance
automatically.

## AMI preparation

Start from the **AWS Deep Learning Base AMI (Amazon Linux 2023)** or any
Amazon Linux 2023 AMI with NVIDIA drivers and CUDA pre-installed. The AMI must
be in the same region as the `AWS_REGION` variable configured in GitHub.

### 1. Launch an instance to build the AMI

```bash
aws ec2 run-instances \
  --image-id ami-XXXXXXXX \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-XXXXXXXX \
  --subnet-id subnet-XXXXXXXX \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=nobrainer-ami-builder}]'
```

### 2. SSH in as ec2-user and configure

```bash
ssh -i your-key.pem ec2-user@<public-ip>
```

All commands below run as `ec2-user`.

#### Install system dependencies

```bash
sudo dnf install -y jq git
```

#### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

#### Create the pre-installed nobrainer venv

The CI workflow expects a directory at `~/nobrainer/` containing a venv with
heavy dependencies (torch, monai, pyro-ppl) already installed. This avoids
re-downloading ~2 GB of packages on every CI run.

```bash
mkdir -p ~/nobrainer
cd ~/nobrainer

uv venv --python 3.14 .venv

# Install the heavy GPU dependencies into the base venv
uv pip install \
  torch \
  monai \
  pyro-ppl \
  pytorch-lightning \
  pytest
```

#### Verify GPU access

```bash
source ~/nobrainer/.venv/bin/activate
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA: {torch.version.cuda}')
print(f'PyTorch: {torch.__version__}')
"
deactivate
```

### 3. Create the AMI

Stop the instance (or use `--no-reboot`), then:

```bash
aws ec2 create-image \
  --instance-id i-XXXXXXXXX \
  --name "nobrainer-pytorch-gpu-$(date +%Y%m%d)" \
  --description "Amazon Linux 2023 + CUDA + PyTorch + uv for nobrainer GPU CI" \
  --no-reboot
```

Note the resulting AMI ID — this goes into the `AWS_IMAGE_ID` variable.

### 4. Terminate the builder instance

```bash
aws ec2 terminate-instances --instance-id i-XXXXXXXXX
```

## GitHub configuration

### Secrets (Settings → Secrets → Actions)

| Name | Description |
|------|-------------|
| `AWS_KEY_ID` | IAM access key with EC2 RunInstances/TerminateInstances/DescribeInstances permissions |
| `AWS_KEY_SECRET` | Corresponding secret access key |
| `GH_TOKEN` | GitHub PAT with `repo` scope (used by machulav/ec2-github-runner to register the runner) |

### Variables (Settings → Variables → Actions)

| Name | Example | Description |
|------|---------|-------------|
| `AWS_REGION` | `us-east-1` | Region where the AMI lives |
| `AWS_IMAGE_ID` | `ami-0abc123def456` | The AMI created above |
| `AWS_INSTANCE_TYPE` | `g4dn.xlarge` | 1x T4 GPU (~$0.53/hr); `p3.2xlarge` for V100 |
| `AWS_SUBNET` | `subnet-0abc123` | Must have internet access for runner registration |
| `AWS_SECURITY_GROUP` | `sg-0abc123` | Allow outbound HTTPS (port 443) |

## IAM policy (minimum permissions)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:CreateTags"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "*"
    }
  ]
}
```

## Updating the base venv

When upgrading PyTorch or other dependencies, SSH into a running instance (or
launch the AMI), update `~/nobrainer/.venv`, and create a new AMI snapshot:

```bash
ssh -i your-key.pem ec2-user@<ip>
cd ~/nobrainer
uv pip install --upgrade torch monai pyro-ppl pytorch-lightning
# Then create a new AMI and update AWS_IMAGE_ID in GitHub variables
```
