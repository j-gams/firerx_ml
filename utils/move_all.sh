aws_dest=$1

# move utils file from utils dir
scp -i ~/aws_pems/jega7451.firerx.pem ../utils/utils.py $aws_dest:~/firerx_ml/utils/
scp -i ~/aws_pems/jega7451.firerx.pem ../utils/firerx_ml_env.yml $aws_dest:~/firerx_ml/utils/

# move various files from manage data
scp -i ~/aws_pems/jega7451.firerx.pem ../manage_data/align_ci_helpers.py $aws_dest:~/firerx_ml/manage_data/
scp -i ~/aws_pems/jega7451.firerx.pem ../manage_data/align_raster.py $aws_dest:~/firerx_ml/manage_data/
scp -i ~/aws_pems/jega7451.firerx.pem ../manage_data/aws_check.sh $aws_dest:~/firerx_ml/manage_data/
scp -i ~/aws_pems/jega7451.firerx.pem ../manage_data/aws_psychic.sh $aws_dest:~/firerx_ml/manage_data/
scp -i ~/aws_pems/jega7451.firerx.pem ../manage_data/create_pyramid_functions.py $aws_dest:~/firerx_ml/manage_data/
scp -i ~/aws_pems/jega7451.firerx.pem ../manage_data/create_pyramid_set.py $aws_dest:~/firerx_ml/manage_data/
scp -i ~/aws_pems/jega7451.firerx.pem ../manage_data/configs/* $aws_dest:~/firerx_ml/manage_data/configs

# move pem file
scp -i ~/aws_pems/jega7451.firerx.pem ~/aws_pems/jega7451.firerx.pem $aws_dest:~/aws_pems
