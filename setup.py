from setuptools import setup, find_packages

setup(
    name="aira",                    
    version="0.1.0",                
    description="AI Research Agent for recommender systems and personalization",
    author="Jong Ha Lee",
    packages=find_packages("src"),
    package_dir={"": "src"},       
    python_requires=">=3.9",
    install_requires=[
    ],
)
