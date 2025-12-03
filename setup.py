from setuptools import setup, find_packages

setup(
    name="aira",                    # 패키지 이름 (pip에서 보이는 이름)
    version="0.1.0",                # 버전은 대충 0.1.0부터 시작
    description="AI Research Agent for recommender systems and personalization",
    author="Jong Ha Lee",
    packages=find_packages("src"),  # src/ 아래의 모든 패키지 탐색
    package_dir={"": "src"},        # 실제 소스코드는 src/ 밑에 있다
    python_requires=">=3.9",
    install_requires=[
        # 여기에는 런타임에 꼭 필요한 것만 넣고,
        # 개발 편하게 하려면 requirements.txt 그대로 써도 됨.
        # 예시:
        # "pydantic",
        # "langchain",
        # "pypdf",
    ],
)
