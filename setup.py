from setuptools import setup, find_packages

requirements = [
    "paddlepaddle",
    "paddleocr>=2.0.1",
    "opencv-python",
    "srt",
]

setup(
    name="wesubtitle_copy",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": [
        "wesubtitle_copy = wesubtitle_copy.main:main",
    ]},
)
