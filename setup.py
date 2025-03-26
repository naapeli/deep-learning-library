from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name='DLL',
    version='1.0',
    author='Aatu Selkee',
    author_email='aatu.selkee@gmail.com',
    description='A basic deep and machine learning library. Made for learning purposes and is not supposed to be used for true applications.',
    # long_description='A detailed description of the project',
    # long_description_content_type='text/markdown',
    url='https://github.com/naapeli/deep-learning-library',
    packages=find_packages(where='DLL'),
    install_requires=read_requirements(),
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: OS Independent',
    # ],
    # python_requires='>=3.6',
)
