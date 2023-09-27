# 코드
https://github.com/tuliotoffolo/wta

# Requirement
cplex
## cplex 설치

- IBM 홈페이지에서 IBM ILOG CPLEX Optimization Studio 설치를 받는다.
- 학생 이메일로 가입하여 무료 설치 가능

Installation on AIX and Linux
The AIX and Linux installers are executable files and, to launch an installation, the file should have execute permission and should be executed from the command line. For example, issue the command ./<installername>.bin, where <installername> is the name of the installer. You can give a file execute permission with the command **chmod u+x <installername>.bin.**

~~On AIX, the default installation location of IBM ILOG CPLEX Optimization Studio is: /opt/IBM/ILOG/CPLEX_Studio[edition]128~~

On Linux, the default installation location of IBM ILOG CPLEX Optimization Studio is: /opt/ibm/ILOG/CPLEX_Studio[edition]128

where [edition] can be: * Community, for the Community Edition * empty, for the standard commercial edition

During installation you may have selected a different location ([install_dir]).

# 환경설정

```bash
conda create -n 230912_milp_wta python=3.9
conda activate 230912_milp_wta
conda config --env --set subdir osx-64
```

`cplex_studio2211.linux_x86_64.bin` 다운로드

언어 선택 등등

``` bash
chmod u+x cplex_studio2211.linux_x86_64.bin
./cplex_studio2211.linux_x86_64.bin 
```

`/opt/ibm/ILOG/CPLEX_Studio2211`
를 기본 경로로 하려고 했더니 권한 없음
그래서 `/ssd2/kawon/230912_IBM_CPLEX`로 설치

```
If you want to use the CPLEX or CP Optimizer engines through their Python 
APIs, you need to tell Python where to find them.  To do so, enter the 
following command into a terminal :
<<<<<<
python /ssd2/kawon/230912_IBM_CPLEX/python/setup.py install
>>>>>>
```

```bash
python /ssd2/kawon/230912_IBM_CPLEX/python/setup.py install
```

conda install numpy

> scipy 최신버전으로 하니 GLIBCXX_3.4.29가 없다는 오류 나옴

conda install scipy=1.6


---

3.6 버전으로 다시

conda create -n 230924_milp_wta python=3.6
conda install numpy
conda install scipy=1.5

해도 여전히 오류가 뜬다.