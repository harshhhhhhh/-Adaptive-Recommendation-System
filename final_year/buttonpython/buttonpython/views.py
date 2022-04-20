from django.shortcuts import render
import sys
from subprocess import run,PIPE


def button(request):
    return render(request,'home.html')
def external(request):
    out= run([sys.executable,'C://Users//Dhruval//final_year//booksrs.py'],shell=False,stdout=PIPE)
    #print(out)

    return render(request,'next.html',{'data1':out.stdout})