#!/bin/bash

# Script to verify files downloaded from CRCNS.org from NERSC
# Version 0.9, Oct. 28, 2014


CHECKSUMS_FILE=checksums.md5

# select one of the following
# OS="mac"
 OS="linux"


function get_hash {
   fn=$1
   if [ "$OS" = "mac" ]; then
      md5_out=`md5 $fn`
      # output looks like:
      # MD5 (file1.txt) = d594185742a01bf7f4464035238cd386
      RE="MD5 \($fn\) = (.*)$"
   elif [ "$OS" = "linux" ]; then
      md5_out=`md5sum $fn`
      # output looks like:
      # 9155a556de6473fd08244fa7d7a901b1  file3.txt
      RE="([^ ]+)  $fn$"
   else
      echo "OS type not found"
      echo "Aborting."
      exit 1
   fi
   if [[ "$md5_out" =~ $RE ]]; then
      hash=${BASH_REMATCH[1]};
   else
      echo "Unable to parse output of md5:"
      echo $md5_out
      echo "Aborting"
      exit 1
   fi
}

function main {
   FIN="$DSID/$CHECKSUMS_FILE"
   if [ ! -e "$FIN" ]; then
      echo "Could not find file '$FIN'.  Aborting."
      exit 1
   fi
   file_count=0
   valid_count=0
   error_count=0
   while read line   
   do
      parts=($line)
      check_hash=${parts[0]}
      fn=${parts[1]}
      path="$DSID/$fn"
      if [ -e "$path" ]; then
         file_count=$[$file_count + 1]
         get_hash $path
         if [[ "$hash" = "$check_hash" ]]; then
            echo "$fn validated."
            valid_count=$[$valid_count + 1]
         else
            echo "** HASH DOES NOT MATCH: $fn"
            error_count=$[$error_count + 1]
         fi
      fi
   done < $FIN
   echo "Processed $file_count files."
   echo "Number validated=$valid_count.  Number hash not matching=$error_count."
   echo "All done."
}


# start of script is here
if [ "$#" -ne 1 ]; then
    read -p 'CRCNS.org dataset ID to verify files? example pvc-1): ' DSID
else
    DSID="$1"
fi
main
exit 0


