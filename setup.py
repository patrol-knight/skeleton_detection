import os
from glob import glob

from setuptools import find_packages, setup


package_name = "skeleton_detection"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="samdong",
    maintainer_email="hydong__sam@outlook.com",
    description="ROS 2 Python package for skeleton detection and tracking.",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "camera_reader_node = skeleton_detection.camera_reader:main",
            "skeleton_detection_node = skeleton_detection.skeleton_detection:main",
        ],
    },
)
