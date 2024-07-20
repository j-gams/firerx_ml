pemdir=$1
#echo "pemdir is $pemdir"
workername=$2
#echo "workername is $workername"
instdir=$3
#echo "instdir is $instdir"
assignstr=$4
#echo "assignstr is $assignstr"
shnumber=$5
#echo "fileext $shnumber"

#echo begin > "localfile${shnumber}.txt"
#ssh -i $pemdir $workername "echo testingagain > al2.txt"

commandstr='"cd ~/'${instdir}'; nohup ~/anaconda3/bin/python align_raster.py core_aws=false core_aw2=true skip_trm=true skip_rrj=true skip_erj=true skip_res=true skip_raw=true pram_sub='${assignstr}' > ~/align_log.txt"'

commandstr2="ssh -i ${pemdir} ${workername} ${commandstr}"

#ssh -i $pemdir $workername "test -f ~/anaconda3/bin/python && echo fileexists1"
#ssh -i $pemdir $workername "cd ~/${instdir}; test -f align_raster.py && echo fileexists2"

eval "$commandstr2"
#ssh -i $pemdir $workername $commandstr

#nohup ssh -i $pemdir $workername '"cd ~/'${instdir}'; nohup ~/anaconda3/bin/python align_raster.py core_aws=false core_aw2=true locl_ras=../'${instdir}'/ locl_tro=./geotifs_raw locl_rjo=geotifs_raw/ locl_clp=aligned_out skip_trm=true skip_rrj=true skip_erj=true skip_res=true skip_raw=true pram_sub='${assignstr}' > ~/align_log.txt"' > "localfile${shnumber}.txt"

echo "verifying3" >> "localfile${shnumber}.txt"

