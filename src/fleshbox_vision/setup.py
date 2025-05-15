from setuptools import find_packages, setup

package_name = 'fleshbox_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='el22mf@leeds.ac.uk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fleshbox_navigator = fleshbox_vision.fleshbox_navigator:main',
            'blood_detection = fleshbox_vision.blood_detection:main',
            'stereo_depth = fleshbox_vision.stereo_depth:main',
            'disparity_viewer = fleshbox_vision.disparity_viewer:main'

        ],
    },
)
