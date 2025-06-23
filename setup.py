from setuptools import setup, find_packages

setup(
    name='DyFilterAttack',
    version='0.1.5',
    packages=find_packages(),  # 自动查找所有包和子包
    install_requires=[
        # 如果有依赖项，可以列在这里
    ],
    include_package_data=True,
)
