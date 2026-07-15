import os
from glob import glob
from setuptools import setup

package_name = 'rubik_control'

setup(
    name=package_name,
    version='0.0.0',
    # --- AQUÍ ESTÁ LA MAGIA: Incluimos explícitamente la subcarpeta ---
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files del proyecto (arranque con un solo comando)
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Antonio Balboa',
    maintainer_email='abalpa19@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orquestador = rubik_control.main:main',
            'robot_node = rubik_control.robot:main',
            'metricas = rubik_control.metricas:main'
        ],
    },
)