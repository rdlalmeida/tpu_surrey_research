#!/bin/bash
# Use the vm_log.txt file to capture all error output from this file. Start each "session" by setting
# the datetime at the top before sending all errors to that file. I need this because the stupid
# Google Cloud console has this stupid habit of closing when an error is raised, destroying all information
# about said error, which makes debugging this crap extremely difficult...
# Create a datetime stamp at the top of the entry
printf "$(date)\n" > vm_log.txt

# And redirect all stderr to that file
2>>vm_log.txt

# Turns out that all commands that I need to use have a common prefix, i. e., the first few terms of the command are always
# the same. Define these as a prefix string and then add the rest accordigly
CMD_PREFIX="gcloud compute tpus tpu-vm"

# Use this nice and simple error handling routine to prevent those nasty console quitting errors!
error_handler() {
    echo "Error ($?): ($1)\n"
	return 1
	
	# echo "Attention: Unable to find a TPU Virtual Machine named '$1'"
}

function join_by() {
	local d=${1-} f=${2-}
	if shift 2; then
		return %s "$f" "${@/#/$d}"
	fi
}

function get_VM_status() {
	# This section used the variables defined (it tests them first, just in case) and determines if there are any 
	# VM's configured. If so, it tries to start it, stop it (asking the user first), or creates a new one if none exist
	if [[ $ZONE = "" ]];
	then
		printf "Unable to determine a valid project zone (\$ZONE not defined)\n"
		return 1
	fi

	# I need the PROJECT_ID also
	if [[ $PROJECT_ID = "" ]];
	then
		printf "Unable to determine a valid project ID (\$PROJECT_ID not defined)\n"
		return 1
	fi

	# And the VM name, which should always be the same
	if [[ $VM_NAME = "" ]];
	then
		printf "Unable to determine a valid Virtual Machine name (\$VM_NAME not defined)\n"
		return 1
	fi

	# If I got to this point, I have a valid ZONE and PROJECT_ID. Continue to determine if there are any VMs associated to it
	# Run this problematic instruction given that it can throw an error. Unfortunatelly, the information I need may be in that error
	# string, so I need to capture anything that is returned from it. To be able to store the error string if one is raised, I've developed
	# that 'error_handler()' routine at the end
	export VM_OUTPUT=$( ${CMD_PREFIX} describe ${VM_NAME} --zone ${ZONE} --project ${PROJECT_ID} 2>&1 ) || error_handler $VM_NAME
	
	# Now I need to run a few 'greps' to extract any of the configured VM states (I got these from trial and error and the stuff in
	# https://cloud.google.com/compute/docs/instances/instance-life-cycle)
	# NOTE: There are way more possible states than the ones defined bellow, but for now, these are the ones to which I have a plan
	# for.
	VM_STATES='STOPPED|STOPPING|READY|RUNNING|NOT_FOUND'

	# Run a simple grep that, if the state string is somewhere in the VM_OUTPUT string, it extracts it to the VM_STATUS variable
	VM_STATUS=$(echo $VM_OUTPUT | grep -o -E $VM_STATES)

	# Check also if none of the states defined in VM_STATES was found in the VM_OUTPUT string. If that is the case,
	# signal this as a new, "UKNOWN_STATE" and update this function at some point
	if [[ $VM_STATUS = "" ]];
	then
		VM_STATUS="UNKNOWN_STATE"
	fi

	# Don't need to do anything else at this point. Bash scripts are a bit different from "regular" programming languages,
	# namely, the variables edited within the scope of this function THAT ARE NOT LIMITED TO IT, i.e., global variables vs.
	# local (function scoped) variables. This means that any alteration to global variables (such as the $VM_STATUS) whitin
	# the scope of this function are actually permanent, i.e., they don't get reverted with the function's completion.
	# Therefore, all I need is to run this function and then simply access the $VM_STATUS variable to determine its result

	# Finish this with a success return
	return 0
}

# This command lists all the projects configured in the logged in account and sets the PROJECT_ID variable to whatever string is between 'PROJECT_ID:' and the next space. Usually this equates to the project_id of the first project configured in the account (This is what whole '| sed ...' command is for
# SAMPLE OUTPUT STRING (for reference)
# PROJECT_ID: surrey-collab-tpu-research01 NAME: Surrey-collab-tpu-research01 PROJECT_NUMBER: 175971125724

# Get the main output for the project definition. NOTE: this script works best, and assumes, a single project in the list, though it can easily be adapted to deal with multiple projects as well
export PROJ_DEF=$(gcloud projects list)

# Extract the PROJECT_ID to a variable with the same name
export PROJECT_ID=$(echo $PROJ_DEF | grep -o -P '(?<=PROJECT_ID: ).*(?= NAME: )')

# The same goes for the project name
export PROJECT_NAME=$(echo $PROJ_DEF | grep -o -P '(?<= NAME:).*(?= PROJECT_NUMBER: )')

# Finally, the project number
export PROJECT_NUMBER=$(echo $PROJ_DEF | grep -o -P '(?<= PROJECT_NUMBER: ).*')

# Print out all these variables for reference
printf "\n\nCurrent Project parameters:\n"
printf "PROJECT_ID = ${PROJECT_ID}\n"
printf "PROJECT_NAME = ${PROJECT_NAME}\n"
printf "PROJECT_NUMBER = ${PROJECT_NUMBER}\n"

printf "\nThese variable were set globally as well:\n"
# Set a number of handy variables to global vars (to make it easier to resume these things)

# NOTE: The following list of zones are valid zones where a v2-8 core TPUs can be provisioned
# This list was obtained from https://cloud.google.com/tpu/docs/regions-zones, which is what
# Google Cloud advises me to consult when they reject starting my TPU VM due to lack of capacity
# Un-comment these zones, one at a time, until one works (there's sufficient TPU capacity in the
# data center to provision the Virtual Machine)

# ------------------------------------- ZONES ------------------------------------------------------
# Europe Zones v2 cores
export ZONE="europe-west4-a"


# US Zones v2 cores
# These zones have v2-8 TPUs
# export ZONE="us-central1-b"
# export ZONE="us-central1-c"
# export ZONE="us-central1-f"		# NOTE: This zone is only available for TPU Research Cloud (TRC) participants.

# This zone has all the remaining core arrangements (32, 128, 256 and 512 cores) for v2 TPUs
# export ZONE="us-central1-a"

# Asia
# Asia is quite limited in this regard. There's only one Zone available for v2 TPUs
# export ZONE="asia-east1-c"
# ------------------------------------- ZONES ------------------------------------------------------

# The ACCELERATOR_TYPE happens to be one of the most important start-up parameters. This flag is composed of two parts split by the '-' caracter. The first element is the version of the TPU units to use and is always preceeded by a 'v' and the second parameter is the number of cores to provision for the requested VM. A 'v3-128' flag request 128 cores of the version 3 of the TPU accelerator and so on. For testing purposes, and to guarantee as best as possible the allocation of these resources, chose low version and a minimal number of cores to start, e. g., 'v2-8'.
# export ACCELERATOR_TYPE="v3-128"
export ACCELERATOR_TYPE="v3-8"
export VERSION="tpu-vm-tf-2.16.1-pjrt"
export SERVICE_ACCOUNT="surrey-tpu-research-srv-acct@surrey-collab-tpu-research06.iam.gserviceaccount.com "
export VM_NAME="surrey-tpu-vm01"
# export VM_NAME="fake_vm_name_for_testing01"

printf "\$ZONE = ${ZONE}\n"
printf "\$ACCELERATOR_TYPE = ${ACCELERATOR_TYPE}\n"
printf "\$VERSION = ${VERSION}\n"
printf "\$SERVICE_ACCOUNT = ${SERVICE_ACCOUNT}\n"
printf "\$VM_NAME = ${VM_NAME}\n"

printf "\n"

# ----------------------------------------------------------------------------------------------------------

# First, turn off the error stopping, i.e., instruct this script to "ignore" any errors in the next instructions. This
# ignoring is limited to the script stopping and exiting the main process. The error still happens and needs to be addressed
set -e

# Run the function to determine the status of the detected VM
get_VM_status || error_handler "Unable to get a valid VM status for '${VM_NAME}'"

# Capture the result into the proper variable
printf "Current \$VM_STATUS = ${VM_STATUS}\n"

# Turn off the error ignoring. From now on, any unexpected errors are going to stop this script
set +e

# Before going into the switch case, define a priori all the commands than might be needed, given that I might need
# to invoke one of them multiple times

# First, the one to create/provisioning a new VM
VM_CREATE="${CMD_PREFIX} create ${VM_NAME} --zone ${ZONE} --accelerator-type ${ACCELERATOR_TYPE} --version ${VERSION} --service-account ${SERVICE_ACCOUNT} --project ${PROJECT_ID}"

# The one to start an already provisioned VM
VM_START="${CMD_PREFIX} start ${VM_NAME} --zone ${ZONE} --project ${PROJECT_ID}"

# The corresponding command to stop a running VM
VM_STOP="${CMD_PREFIX} stop ${VM_NAME} --zone ${ZONE} --project ${PROJECT_ID}"

# And the one to connect to a running VM using SSH
VM_CONN="${CMD_PREFIX} ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT_ID}"

# One last command to delete a provisioned VM, though, for now, this option is not available yet
VM_DELETE="${CMD_PREFIX} delete ${VM_NAME} --zone ${ZONE} --project ${PROJECT_ID}"

# Now run a switch case for every one of the states defined so far and add the logic to execute in each
case $VM_STATUS in
	"STOPPED")
	printf "TPU VM '${VM_NAME}' in zone '${ZONE}' is currently '${VM_STATUS}'\n"
	printf "Do you wish to start it? (Y/n)\n"
	read answer

	case $answer in
		"Y" | "y" | "Yes" | "yes" | "YES")
			
			printf "Starting '${VM_NAME}' in zone '${ZONE}' from project ${PROJECT_ID}...\n"
			printf "Running '${VM_START}'...\n"

			# Run the start VM command while looking for potential errors
			$VM_START || error_handler "Unable to start VM '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'\n"

			# If the script gets here, I have an online TPU VM. Now I need to connect to it by SSH

			printf "TPU VM '${VM_NAME}' is online. Do you wish to connect to it using SSH? (Y/n)\n"
			read reply

			case $reply in
				"Y" | "y" | "Yes" | "yes" | "YES")

					printf "Connecting to '${VM_NAME}' in zone '${ZONE}' using SSH...\n"
					printf "${VM_CONN}\n"
					$VM_CONN || error_handler "Unable to connect via SSH to '${VM_NAME}'\n"

				;;	# End of Y | y | Yes | yes | YES case

				"N" | "n" | "No" | "no" | "NO")
					
					printf "OK!\n"
					return 0

				;;	# End of N | n | No | no | NO case

				*)
					printf "Unknow option: '${reply}'. Exiting\n"
					return 1
				;;	# End of default case
			esac
			# -------------------------------------------------------------------------------------
			;;	# End of Y | y | Yes | yes | YES case

		"N" | "n" | "No" | "no" | "NO")
			# The VM is stopped. If the user does not want to connect to it, check if he/she wants to delete it
			printf "TPU VM '${VM_NAME}' in zone '${ZONE}' is still '${VM_STATUS}'\n"
			printf "Do you wish to delete it? (Y/n)\n"
			read answer

			case $answer in
				"Y" | "y" | "Yes" | "yes" | "YES")
					# Run the delete command
					printf "Deleting VM '${VM_NAME}' in zone '${ZONE}' from project '${PROJECT_ID}'\n" 
					printf "Using '${VM_DELETE}...\n'"
					$VM_DELETE || error_handler "Unable to delete VM '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'\n"
					printf "Deleted VM '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'\n"

				;;		# End of Yes case

				"N" | "n" | "No" | "no" | "NO")

					printf "OK! VM still in storage.\n"
					return 0

				;;		# End of No case

				*)

					printf "Unknown option: '${answer}'. Exiting\n"
					return 1

				;;		# End of default case
			esac
		;;	# End of N | n | No | no | NO case

		*)
			printf "Unkown option: '${answer}'. Exiting\n"
			return 1
		;;	# End of default case
	esac

	;;		# End of STOPPED case

	"STOPPING")

		# There's not a lot to do in this case other than waiting... The VM is halfway between RUNNING and STOPPED
		printf "VM '${VM_NAME}' in zone '${ZONE}' is currently ${VM_STATUS}. Wait a few seconds and try again please\n"
		return 0
	;;		# End of STOPPING case

	"READY")
		# The READY state means the VM is online and ready to be connected to via SSH
		printf "TPU VM '${VM_NAME}' in zone '${ZONE}' is currently '${VM_STATUS}'\n"
		printf "Do you wish to connect to it via SSH? (Y/n)\n"
		read answer

		case $answer in
			"Y" | "y" | "Yes" | "yes" | "YES")
				
				printf "Connecting to '${VM_NAME}' in zone '${ZONE}' using SSH...\n"
				printf "${VM_CONN}\n"
				$VM_CONN || error_handler "Unable to connect via SSH to '${VM_NAME}'\n"

			;;	# End of Y | y | Yes | yes | YES case

			"N" | "n" | "No" | "no" | "NO")

				# If the user does not wants to connect to the VM, check if he/she wants to stop it
				printf "TPU VM '${VM_NAME}' in zone '${ZONE}' from project '${PROJECT_ID}' is still ${VM_STATUS}\n"
				printf "Do you wish to stop it? (Y/n)\n"
				read reply

				case $reply in
				"Y" | "y" | "Yes" | "yes" | "YES")

					printf "Stopping '${VM_NAME}' in zone '${ZONE}' from project ${PROJECT_ID}...\n"
					printf "Running '${VM_STOP}'...\n"

					# Run the stop VM command while looking for potential errors
					$VM_STOP || error_handler "Unable to stop VM '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'\n"

					# Once the VM has been stopped, ask also if the user wants to delete it to free up its resources
					printf "TPU VM '${NAME}' is stopped. Do you wish to delete it to free up resources? (Y/n)\n"
					read ans

					case $ans in
						"Y" | "y" | "Yes" | "yes" | "YES")

							printf "Deleting VM '${VM_NAME}' in zone '${ZONE}' from project '${PROJECT_ID}'\n"
							printf "Using '${VM_DELETE}...\n'"
							$VM_DELETE || error_handler "Unable to delete VM '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'\n"
							printf "Deleted VM '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'\n"

						;;	# End of Yes case

						"N" | "n" | "No" | "no" | "NO")

							printf "OK! VM image was not deleted.\n"
							return 0

						;;	# End of No case

						*)

							printf "Unknown option: '${ans}'. Exiting\n"
							return 1

						;;	# End of default case
					esac
				;;	# End of Yes case

				"N" | "n" | "No" | "no" | "NO")
					printf "OK! Nothing to do\n"
					return 0
				;;	# End of No case

				*)
					printf "Unknown option: '${reply}'. Exiting\n"
					return 1

				;;	# End of the default case
			esac

			;;	# End of N | n | No | no | NO case

			*)

				printf "Unknown option: '${answer}'. Exiting\n"
				return 1

			;;	# End of default case
		esac
	;;		# End of READY case

	"RUNNING")

		# The VM is up and running. From here I can either connect to it (via ssh) or stop the VM
		printf "VM '${VM_NAME}' in zone '${ZONE}' is currently ${VM_STATUS}.\n"
		printf "Do you wish to connect to it via SSH? (Y/n)\n"
		read conn

		case $conn in
			"Y" | "y" | "Yes" | "yes" | "YES")

				printf "Connecting to '${VM_NAME}' in zone '${ZONE}' using SSH...\n"
				printf "${VM_CONN}\n"
				$VM_CONN || error_handler "Unable to SSH connect to VM '${VM_NAME}'\n"

			;;	# End of the Y | y | Yes | yes | YES
			"N" | "n" | "No" | "no" | "NO")
				
				printf "OK!\n"
				return 0

			;;	# End of the N | n | No | no | NO case
			*)
				printf "Unknown option: '${conn}'. Exiting\n"
				return 1
			;;	# End of default case
		esac

	;;		# End of the RUNNING case
	"NOT_FOUND")
		printf "No TPU VMs were provisioned yet for project '${PROJECT_ID}' for zone '${ZONE}'. Do you wish to create one? (Y/n)\n"
		read create

		case $create in
			"Y" | "y" | "Yes" | "yes" | "YES")
				printf "Provisioning a new TPU VM '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'...\n"
				printf "${VM_CREATE}\n"
				$VM_CREATE || error_handler "Unable to provision a new VM for zone '${ZONE}'\n"

				printf "New TPU VM provisioned with name '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'\n"
				printf "Do you wish to connect to it via SSH? (Y/n)\n"
				read conn_ans

				case $conn_ans in
					"Y" | "y" | "Yes" | "yes" | "YES")
						
						printf "Connecting to '${VM_NAME}' in zone '${ZONE}' using SSH...\n"
						printf "${VM_CONN}\n"
						$VM_CONN || error_handler "Unable to connect using SSH to VM '${VM_NAME}'\n"

					;;	# End of "Y" | "y" | "Yes" | "yes" | "YES" case

					"N" | "n" | "No" | "no" | "NO")
						
						printf "OK!\n"
						return 0

					;;	# End of "N" | "n" | "No" | "no" | "NO" case

					*)

						printf "Unknown option: '${conn_ans}'. Exiting\n"
						return 1

					;;	# End of the default case
				esac

			;;	# End of "Y" | "y" | "Yes" | "yes" | "YES" case

			"N" | "n" | "No" | "no" | "NO")
				printf "OK!\n"
				return 0
			;;	# End of "N" | "n" | "No" | "no" | "NO" case

			*)
				printf "Unknown option: '${create}'. Exiting\n"
				return 1
			;;	# End of the default case
		
		esac

	;;		# End of the NOT_FOUND case
	"UNKNOWN_STATE")
		printf "Unable to determine a state for TPU VM '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'\n"
		printf "Please review the project parameters to solve this issue. Unable to continue...\n"
		return 1
	;;		# End of the UNKNOWN_STATE

	*)
		printf "Unable to determine the status of VM '${VM_NAME}' in zone '${ZONE}' for project '${PROJECT_ID}'\n"
		return 1
	;;		# End of the default case

esac

# Close the log session by printing this line at the bottom, so that I can identify all individual executions
# of this script
printf "---------------------------------------------------------------------------------------------\n" >> vm_log.txt