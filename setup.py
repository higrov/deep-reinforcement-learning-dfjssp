from setuptools import setup, find_packages

setup(name='gym_cooking',
      version='0.0.1',
      description='Too Many Cooks: Overcooked environment',
      author='Rose E. Wang',
      email='rewang@stanford.edu',
      packages=find_packages(),
      install_requires=[
            'cloudpickle==1.3.0',
            'decorator==4.4.2',
            'dill==0.3.2',
            'future==0.18.2',
            'gym==0.17.2',
            'matplotlib==3.3.2',
            'networkx==2.5',
            'numpy==1.23.2',
            'pandas==1.1.5',
            'Pillow>=8.1.1',
            'pygame==2.1.0',
            'pyglet==1.5.0',
            'pyparsing==2.4.7',
            'python-dateutil==2.8.2',
            'pytz==2022.2.1',
            'scipy==1.9.0',
            'seaborn==0.11.0',
            'six==1.16.0',
            'termcolor==1.1.0',
            'tqdm==4.64.1',
            'rich==12.6.0',
            # SEAC STUFF
            #'torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116',
            'stable-baselines3==1.0',
            'werkzeug==2.2.2',
            # MAPPO STUFF
            'tensorboardX==2.5.1',
            # Others
            'pyyaml==6.0',
            'tensorboard==2.11.0',
            'wandb==0.13.9',
]
      )
