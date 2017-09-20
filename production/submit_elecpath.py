import os

for i in range(1, 101):
    ib = str(i)
    os.system('cp skeleton train_brick'+ib+'.sh')
    os.system('echo "python Predict_electron_path.py -b '+ib+' -t" >> train_brick'+ib+'.sh')
    os.system('qsub train_brick'+ib+'.sh')

    os.system('cp skeleton test_brick'+ib+'.sh')
    os.system('echo "python Predict_electron_path.py -b '+ib+'" >> test_brick'+ib+'.sh')
    os.system('qsub test_brick'+ib+'.sh')

exit()
#for i in range(1, 101):
#    ib = str(i)
#    os.system('cp skeleton train_brick'+ib+'.sh')
#    os.system('echo "python add_electron_path.py -b '+ib+' -t" >> train_brick'+ib+'.sh')
#    os.system('qsub -l nodes=1:ppn=4 train_brick'+ib+'.sh')
#
#    os.system('cp skeleton test_brick'+ib+'.sh')
#    os.system('echo "python add_electron_path.py -b '+ib+'" >> test_brick'+ib+'.sh')
#    os.system('qsub -l nodes=1:ppn=4 test_brick'+ib+'.sh')



