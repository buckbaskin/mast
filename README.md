# mast

Tools for discovering and recommendating new accounts to follow on Mastodon

<img alt="Square Rigger Sailing Vessel with five square sails and four jibs on a blue background." src="logo.png" width="250" height="250">

## Example Workflow

### Setup

```
git clone https://github.com/buckbaskin/mast.git
cd mast/
```

```
python3 -m virtualenv v/
source v/bin/activate
pip install -r requirements.txt
```

### Source a batch of samples

```
python3 mast.py download -t 1000
```

### Explore options by pseudo-random sampling and rating content

```
python3 mast.py bandit --explore -t 20
```

### Explore options by exploring content "near" to liked content

```
python3 mast.py bandit --positive -t 20
```

### View the results

```
python3 mast.py report
```
