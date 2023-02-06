#!/bin/bash
#run the FL simulation multiple times to present aggregated test evaluation performance metrics
for i in 1 2 3
do
   	mkdir "rep$i"
	python FL_script.py $i
	
done

