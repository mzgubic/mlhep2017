import os

for i in range(1,101):
    ib = str(i)
    os.system('cp skeleton train_brick'+ib+'.sh')
    os.system('echo "python add_grid.py -b '+ib+' -t" >> train_brick'+ib+'.sh')
    os.system('qsub train_brick'+ib+'.sh')

    os.system('cp skeleton test_brick'+ib+'.sh')
    os.system('echo "python add_grid.py -b '+ib+'" >> test_brick'+ib+'.sh')
    os.system('qsub test_brick'+ib+'.sh')


