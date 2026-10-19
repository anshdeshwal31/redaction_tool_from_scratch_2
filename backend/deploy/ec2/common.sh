#!/usr/bin/env bash
# Shared helpers for the mlredact-dev EC2 scripts.  Sourced by them, not run directly.
#
# Defaults follow MODELS.md section 2 (development instance).  Override any of them with an
# environment variable, e.g.  INSTANCE_TYPE=m7i.2xlarge ./rebuild-instance.sh

# Git Bash on Windows rewrites arguments that look like POSIX paths (/dev/sda1); turn that off.
export MSYS_NO_PATHCONV=1

REGION="${AWS_REGION:-ap-south-1}"
NAME_TAG="${NAME_TAG:-mlredact-dev}"
INSTANCE_TYPE="${INSTANCE_TYPE:-m7i.xlarge}"    # one fixed type for every benchmark run
ROOT_GB="${ROOT_GB:-50}"                        # gp3, encrypted
AMI_NAME_PREFIX="${AMI_NAME_PREFIX:-mlredact-dev-}"

log() { printf '==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# aws wrapper: fixed region, no pager, CRLF stripped (aws.exe emits CRLF under Git Bash).
a() { command aws --region "$REGION" --no-cli-pager "$@" | tr -d '\r'; }

require_aws() {
  command -v aws >/dev/null 2>&1 || die "AWS CLI not found. Install it (winget install Amazon.AWSCLI) and reopen the terminal."
  local who
  who=$(a sts get-caller-identity --query Arn --output text 2>/dev/null) \
    || die "AWS credentials missing or expired. Run 'aws login' (or 'aws configure') and retry."
  log "AWS identity: $who  (region $REGION)"
}

# Sets INSTANCE to the id of the instance tagged Name=$NAME_TAG, or "" if there is none.
# INSTANCE_ID in the environment overrides the lookup.
find_instance() {
  if [[ -n "${INSTANCE_ID:-}" ]]; then INSTANCE="$INSTANCE_ID"; return; fi
  local ids
  ids=$(a ec2 describe-instances \
    --filters "Name=tag:Name,Values=$NAME_TAG" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  (( $(wc -w <<<"$ids") <= 1 )) \
    || die "More than one instance is tagged Name=$NAME_TAG ($ids). Re-run with INSTANCE_ID=<id>."
  INSTANCE="$ids"
}

# inst_field <JMESPath relative to the instance>  ->  text ("None" when absent)
inst_field() {
  a ec2 describe-instances --instance-ids "$INSTANCE" \
    --query "Reservations[0].Instances[0].$1" --output text
}

# Prints the id of the volume attached at the root device, or "None" if it is gone.
root_volume_id() {
  local dev
  dev=$(inst_field RootDeviceName)
  inst_field "BlockDeviceMappings[?DeviceName=='$dev'].Ebs.VolumeId | [0]"
}

wait_ready() {
  log "Waiting for $INSTANCE to be running..."
  a ec2 wait instance-running --instance-ids "$INSTANCE"
  log "Waiting for status checks to pass (about 2-3 minutes)..."
  a ec2 wait instance-status-ok --instance-ids "$INSTANCE"
}

show_connect_info() {
  local dns key
  dns=$(inst_field PublicDnsName)
  key=$(inst_field KeyName)
  cat <<EOF

Instance $INSTANCE ($(inst_field InstanceType)) is up.
The public address changes on every start; update HostName in ~/.ssh/config (MODELS.md section 3):

  Host mlredact-dev
      HostName $dns
      User ubuntu
      IdentityFile ~/.ssh/$key.pem

EOF
}
