import os

for i in range(60, 71):
    ib = str(i)
    os.system('cp skeleton train_brick'+ib+'.sh')
    os.system('echo "python add_tracking.py -b '+ib+' -t" >> train_brick'+ib+'.sh')
    os.system('qsub train_brick'+ib+'.sh -q short')

    os.system('cp skeleton test_brick'+ib+'.sh')
    os.system('echo "python add_tracking.py -b '+ib+'" >> test_brick'+ib+'.sh')
    os.system('qsub test_brick'+ib+'.sh -q short')


