fname="results.txt"
if [ -f fname ] then
	rm $fname

for ((i=30;i<70;i+=10));do
#	echo $i
    python project1.py $i $fname
done
