from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    version="1.0",
    name="sdde",
    url="https://github.com/stdereka/ded-ood",
    author="Stanislav Dereka",
    author_email="st.dereka@gmail.com",
    description='This package provides an implementation of the diversity losses from the paper\
    "Diversifying Deep Ensembles: A Saliency Map Approach for Enhanced OOD Detection, Calibration, and Accuracy".',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=["sdde"],
    package_dir={"sdde": "openood/trainers/diversity/"},
    install_requires=[
        "torch>=1.12.0",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
