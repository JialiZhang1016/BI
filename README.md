# 张佳莉 Machine learning and BI

### git常用命令
- 推送至远程仓库
```
git branch -M master
git remote add origin git@github.com:JialiZhang1016/***.git
git push -u origin master
git push origin master
```

- 添加，提交，删除文档  
```
git status  
git add **  
git commit -m "***"
rm **  
```

- 创建文件夹，切换目录
```
mkdir **  
cd **  
pwd  
```

- 初始化Repository, 查看当前目录下文件  
```
git init  
ls -ah  
ll  
```

- 查看历史文件  
```
git log  
git diff  
```

### Anaconda常用命令
```
conda env list
conda info -e
conda create --name envname python=3.6
conda activate envname
deactivate
conda remove -n envname --all

pip install package -i https://pypi.tuna.tsinghua.edu.cn/simple
```