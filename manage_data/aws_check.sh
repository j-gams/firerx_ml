pemdir=$1
instname=$2
checkloc=$3

commandstr="ssh -i ${pemdir} ${instname} 'test -e ${checkloc}'"

echo $commandstr

if eval "$commandstr"; 
	then
		echo "exists"
	else
		echo "nofile"
fi
