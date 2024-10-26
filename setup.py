from setuptools import setup, find_packages

setup(
    name='agi',  # 包的名称
    version='0.1.0',  # 包的版本
    packages=find_packages(),  # 自动找到所有的包
    description='A simple example module',  # 简短描述
    long_description=open('README.md').read(),  # 从 README 文件读取长描述
    long_description_content_type='text/markdown',  # 长描述的格式
    author='neo',  # 作者姓名
    author_email='guojingneo1988@gmail.com',  # 作者电子邮件
    url='https://github.com/neoguojing/agi',  # 项目网址
    install_requires=[  # 依赖的包
        # 'package_name>=1.0.0',
    ],
    classifiers=[  # 包的分类信息
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 指定 Python 版本要求
)
