#!/bin/bash
# Added a stupid comment to test the github crap
if [[ $VM_NAME = "" ]];
then
	printf "Cannot find a working TPU VM to stop (\$VM_NAME not defined).\n"
	set -e
elif [[ $ZONE = "" ]];
then
	printf "Unable to find a valid zone for the TPU VM (\$ZONE not defined).\n"
	set -e
elif [[ $PROJECT_ID = "" ]];
then
	print "Unable to find a valid project id for the TPU VM (\$PROJECT_ID not defined).\n"
else
	printf "Deleting virtual machine '${VM_NAME}' in zone '${ZONE}' for project id ${PROJECT_ID}\n"
	VM_DELETE="gcloud compute tpus tpu-vm delete ${VM_NAME} --zone ${ZONE} --project ${PROJECT_ID}"
	printf "${VM_DELETE}"
	$VM_DELETE
	printf "Done!\n"
fi
