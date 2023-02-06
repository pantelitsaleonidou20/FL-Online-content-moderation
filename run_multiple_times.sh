#!/bin/bash
for i in 1 2 3
do
   	mkdir "rep$i"
	python test.py $i
	
done

