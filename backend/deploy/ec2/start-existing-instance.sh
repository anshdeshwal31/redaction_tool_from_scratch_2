#!/usr/bin/env bash
# Start the stopped mlredact-dev instance whose root EBS volume still exists.
#
#   bash deploy/ec2/start-existing-instance.sh
#
# If the root volume is gone the instance cannot boot; the script says so and points you at
# rebuild-instance.sh.  Nothing is created or deleted here.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_aws
find_instance
[[ -n "$INSTANCE" ]] || die "No instance tagged Name=$NAME_TAG in $REGION. Run deploy/ec2/rebuild-instance.sh."

state=$(inst_field State.Name)
type=$(inst_field InstanceType)
log "Found $INSTANCE ($type), state: $state"
[[ "$type" == "$INSTANCE_TYPE" ]] \
  || log "WARNING: type is $type, MODELS.md says $INSTANCE_TYPE; benchmark outputs are only comparable on one CPU family."

vol=$(root_volume_id)
if [[ -z "$vol" || "$vol" == "None" ]]; then
  echo "The root volume of $INSTANCE has been deleted, so it cannot start." >&2
  echo "Run deploy/ec2/rebuild-instance.sh to replace it." >&2
  exit 2
fi
log "Root volume: $vol"

case "$state" in
  running)  log "Already running." ;;
  pending)  ;;
  stopping) log "Waiting for it to finish stopping..."
            a ec2 wait instance-stopped --instance-ids "$INSTANCE"
            a ec2 start-instances --instance-ids "$INSTANCE" >/dev/null ;;
  stopped)  log "Starting..."
            a ec2 start-instances --instance-ids "$INSTANCE" >/dev/null ;;
  *)        die "Instance is in state '$state'; cannot start it." ;;
esac

wait_ready
show_connect_info
