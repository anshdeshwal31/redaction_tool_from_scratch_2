#!/usr/bin/env bash
# Replace the mlredact-dev instance whose root EBS volume was deleted.
#
#   bash deploy/ec2/rebuild-instance.sh [--dry-run] [-y]
#
#   --dry-run  print the plan and stop; nothing is created
#   -y         skip the confirmation prompts
#
# What it does:
#   1. Finds the broken instance (Name=mlredact-dev) and refuses to touch it if it still has a
#      root volume (use start-existing-instance.sh then).
#   2. Picks the newest of your own AMIs named mlredact-dev-*; if there is none, the newest
#      Ubuntu 26.04 LTS (else 24.04) from Canonical.
#   3. Launches the instance per MODELS.md: m7i.xlarge, 50 GiB gp3 encrypted root, IMDSv2 required,
#      public IP, SSH (22) open only to your current IP, no IAM role.
#   4. Waits until it is reachable, renames the old instance and (after confirmation) terminates it.
#
# A fresh Ubuntu AMI has none of your setup: repeat MODELS.md section 3 on it, then save an AMI
# named mlredact-dev-YYYYMMDD so the next rebuild starts from it.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SG_NAME="${SG_NAME:-mlredact-dev-ssh}"
UBUNTU_OWNER=099720109477   # Canonical
DRY_RUN=0 YES=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY_RUN=1 ;;
    -y|--yes)   YES=1 ;;
    *)          die "Unknown argument: $arg" ;;
  esac
done

confirm() {   # confirm <prompt>
  (( YES )) && return 0
  local reply; read -r -p "$1 [y/N] " reply
  [[ "$reply" == [yY] ]]
}

newest_ami() {   # newest_ami <owner> <name pattern>  ->  ami id or "None"
  a ec2 describe-images --owners "$1" \
    --filters "Name=name,Values=$2" "Name=state,Values=available" "Name=architecture,Values=x86_64" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text
}

require_aws

# --- 1. the broken instance -----------------------------------------------------------------
find_instance
OLD="$INSTANCE"
KEY="${KEY_NAME:-}"
if [[ -n "$OLD" ]]; then
  vol=$(root_volume_id)
  if [[ -n "$vol" && "$vol" != "None" ]]; then
    die "$OLD still has its root volume ($vol). Use start-existing-instance.sh instead."
  fi
  log "Old instance $OLD ($(inst_field State.Name)) has no root volume."
  SUBNET=$(inst_field SubnetId)
  VPC=$(inst_field VpcId)
  [[ -n "$KEY" ]] || { KEY=$(inst_field KeyName); [[ "$KEY" != "None" ]] || KEY=""; }
else
  log "No existing instance tagged Name=$NAME_TAG; creating a new one."
  read -r SUBNET VPC < <(a ec2 describe-subnets --filters Name=default-for-az,Values=true \
    --query 'Subnets[0].[SubnetId,VpcId]' --output text)
  [[ "$SUBNET" != "None" ]] || die "No default VPC in $REGION. Re-run with SUBNET_ID and VPC_ID set."
fi
SUBNET="${SUBNET_ID:-$SUBNET}"
VPC="${VPC_ID:-$VPC}"
KEY="${KEY:-$NAME_TAG}"

# --- 2. image and disk ----------------------------------------------------------------------
AMI="${AMI_ID:-$(newest_ami self "${AMI_NAME_PREFIX}*")}"
AMI_SRC="your saved AMI"
if [[ "$AMI" == "None" ]]; then
  AMI_SRC="Ubuntu 26.04 LTS"
  AMI=$(newest_ami "$UBUNTU_OWNER" 'ubuntu/images/hvm-ssd-gp3/ubuntu-*-26.04-amd64-server-*')
  if [[ "$AMI" == "None" ]]; then
    log "Ubuntu 26.04 is not published in $REGION; falling back to 24.04."
    AMI_SRC="Ubuntu 24.04 LTS"
    AMI=$(newest_ami "$UBUNTU_OWNER" 'ubuntu/images/hvm-ssd-gp3/ubuntu-*-24.04-amd64-server-*')
  fi
  [[ "$AMI" != "None" ]] || die "Could not find an Ubuntu AMI in $REGION."
fi
ROOT_DEV=$(a ec2 describe-images --image-ids "$AMI" --query 'Images[0].RootDeviceName' --output text)
AMI_GB=$(a ec2 describe-images --image-ids "$AMI" \
  --query "Images[0].BlockDeviceMappings[?DeviceName=='$ROOT_DEV'].Ebs.VolumeSize | [0]" --output text)
SIZE=$(( AMI_GB > ROOT_GB ? AMI_GB : ROOT_GB ))   # a volume can't be smaller than its AMI

# --- 3. network -----------------------------------------------------------------------------
MY_CIDR="${SSH_CIDR:-$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')/32}"
SG=$(a ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text)

cat <<EOF

Plan ($REGION)
  image      $AMI  ($AMI_SRC)
  type       $INSTANCE_TYPE
  root disk  $SIZE GiB gp3, encrypted, $ROOT_DEV, deleted with the instance
  subnet     $SUBNET  (vpc $VPC), public IP on
  key pair   $KEY  $(a ec2 describe-key-pairs --key-names "$KEY" >/dev/null 2>&1 && echo "(exists)" || echo "(will be created, saved to ~/.ssh/$KEY.pem)")
  firewall   $SG_NAME $([[ "$SG" == "None" ]] && echo "(will be created)" || echo "($SG)"): SSH from $MY_CIDR only
  metadata   IMDSv2 required, no IAM role
  old        ${OLD:-none}$([[ -n "$OLD" ]] && echo " -> renamed ${NAME_TAG}-old, then terminated")
  cost       about \$0.21/hour while running (m7i.xlarge, Mumbai) plus about \$4/month for the disk

EOF
(( DRY_RUN )) && { log "Dry run: nothing created."; exit 0; }
confirm "Create this instance?" || die "Aborted."

# --- 4. create ------------------------------------------------------------------------------
if ! a ec2 describe-key-pairs --key-names "$KEY" >/dev/null 2>&1; then
  pem="$HOME/.ssh/$KEY.pem"
  [[ ! -e "$pem" ]] || die "$pem exists but key pair '$KEY' does not exist in AWS. Set KEY_NAME to another name."
  mkdir -p "$HOME/.ssh"
  a ec2 create-key-pair --key-name "$KEY" --query KeyMaterial --output text > "$pem"
  chmod 400 "$pem"
  log "Created key pair '$KEY', private key saved to $pem"
fi

if [[ "$SG" == "None" ]]; then
  SG=$(a ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC" \
    --description "mlredact-dev: SSH from the developer IP only (MODELS.md)" \
    --query GroupId --output text)
  log "Created security group $SG"
fi
if ! a ec2 describe-security-groups --group-ids "$SG" \
     --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text \
     | tr '\t' '\n' | grep -qx "$MY_CIDR"; then
  a ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MY_CIDR" >/dev/null
  log "Allowed SSH from $MY_CIDR (remove old rules for other IPs when you stop work)"
fi

BDM="[{\"DeviceName\":\"$ROOT_DEV\",\"Ebs\":{\"VolumeSize\":$SIZE,\"VolumeType\":\"gp3\",\"Encrypted\":true,\"DeleteOnTermination\":true}}]"
log "Launching..."
NEW=$(a ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$KEY" \
  --block-device-mappings "$BDM" \
  --network-interfaces "DeviceIndex=0,SubnetId=$SUBNET,AssociatePublicIpAddress=true,Groups=$SG" \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME_TAG}]" \
                       "ResourceType=volume,Tags=[{Key=Name,Value=$NAME_TAG}]" \
  --query 'Instances[0].InstanceId' --output text)
log "Launched $NEW"

# Free the Name tag so start-existing-instance.sh finds only the new instance.
[[ -z "$OLD" ]] || a ec2 create-tags --resources "$OLD" --tags "Key=Name,Value=${NAME_TAG}-old"

INSTANCE="$NEW"
wait_ready
show_connect_info

# --- 5. retire the old instance -------------------------------------------------------------
if [[ -n "$OLD" ]]; then
  if confirm "Terminate the old, unbootable instance $OLD?"; then
    a ec2 terminate-instances --instance-ids "$OLD" >/dev/null
    log "Terminated $OLD"
  else
    log "Left $OLD in place (renamed ${NAME_TAG}-old)."
  fi
fi

cat <<EOF
Next: repeat MODELS.md section 3 on the new instance if it was built from plain Ubuntu, then save
an AMI so the next rebuild is quick:
  aws ec2 create-image --region $REGION --instance-id $NEW --name ${AMI_NAME_PREFIX}\$(date +%Y%m%d)
EOF
