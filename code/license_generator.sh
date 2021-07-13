#!/bin/bash
fullCopyright="# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"
copyright="# Copyright (c) Facebook, Inc. and its affiliates."

for i in *.sh *.py */*.sh */*.py */*/*.sh */*/*.py; do
	if grep -q "$copyright" $i; then
		echo "Copyright found in $i"
	else
		echo "Inserting copyright in $i"
		if [ "`head -c 2 $i`" = "#!" ]; then
			# shebang
			head -1 $i >> $i.new
			echo "$fullCopyright" >> $i.new
			tail -n +2 $i >> $i.new
		else
			echo "$fullCopyright" >> $i.new
			cat $i >> $i.new
		fi
		mv $i.new $i
	fi
done
