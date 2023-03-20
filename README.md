## Test results

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>iou</th>
      <th>iou_group_0</th>
      <th>iou_group_1</th>
      <th>iou_group_2</th>
      <th>iou_group_3</th>
      <th>iou_group_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CE+Dice, Unet++ (resnet34) (26.1M)</td>
      <td>87.80</td>
      <td>86.25</td>
      <td>93.53</td>
      <td>94.19</td>
      <td>94.12</td>
      <td>95.88</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CE+Boundary, Unet++ (resnet34) (26.1M)</td>
      <td>85.54</td>
      <td>84.33</td>
      <td>91.52</td>
      <td>87.37</td>
      <td>89.17</td>
      <td>94.67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CE+Dice, Unet++ (resnet101) (68.0M)</td>
      <td>87.82</td>
      <td>86.53</td>
      <td>93.12</td>
      <td>92.56</td>
      <td>91.66</td>
      <td>97.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CE+Boundary, Unet++ (resnet101) (68.0M)</td>
      <td>87.11</td>
      <td>85.61</td>
      <td>92.41</td>
      <td>93.63</td>
      <td>93.73</td>
      <td>94.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CE+Dice, Segformer (nvidia/mit-b0) (3.71M)</td>
      <td>81.68</td>
      <td>79.66</td>
      <td>88.66</td>
      <td>90.57</td>
      <td>90.63</td>
      <td>94.35</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CE+Boundary, Segformer (nvidia/mit-b0) (3.71M)</td>
      <td>84.41</td>
      <td>83.46</td>
      <td>87.53</td>
      <td>88.69</td>
      <td>88.82</td>
      <td>90.96</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>dice</th>
      <th>dice_group_0</th>
      <th>dice_group_1</th>
      <th>dice_group_2</th>
      <th>dice_group_3</th>
      <th>dice_group_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CE+Dice, Unet++ (resnet34) (26.1M)</td>
      <td>93.33</td>
      <td>92.44</td>
      <td>96.65</td>
      <td>96.99</td>
      <td>96.91</td>
      <td>97.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CE+Boundary, Unet++ (resnet34) (26.1M)</td>
      <td>91.74</td>
      <td>91.02</td>
      <td>95.55</td>
      <td>92.27</td>
      <td>93.90</td>
      <td>97.26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CE+Dice, Unet++ (resnet101) (68.0M)</td>
      <td>93.14</td>
      <td>92.35</td>
      <td>96.41</td>
      <td>96.01</td>
      <td>95.32</td>
      <td>98.54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CE+Boundary, Unet++ (resnet101) (68.0M)</td>
      <td>92.91</td>
      <td>92.03</td>
      <td>96.04</td>
      <td>96.69</td>
      <td>96.75</td>
      <td>97.32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CE+Dice, Segformer (nvidia/mit-b0) (3.71M)</td>
      <td>89.12</td>
      <td>87.76</td>
      <td>93.89</td>
      <td>95.01</td>
      <td>95.08</td>
      <td>97.09</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CE+Boundary, Segformer (nvidia/mit-b0) (3.71M)</td>
      <td>91.16</td>
      <td>90.59</td>
      <td>92.94</td>
      <td>93.88</td>
      <td>94.05</td>
      <td>95.27</td>
    </tr>
  </tbody>
</table>

## Reproduce the results

### Using python3.9 environment (recomended)

Setup environment and install all needed requirements

```sh
python3.9 -m venv .kvasir_seg_venv
```
```sh
source .kvasir_seg_venv/bin/activate
```
```sh
git clone https://github.com/GerasimovIV/kvasir-seg.git
```
```sh
cd kvasir-seg
```
```sh
pip install -r requirements.txt
```
Download and extarct datasets
```sh
make datasets
```

now you are ready to use any scripts or notebook 🚀. For example in you may use [test notebook](https://github.com/GerasimovIV/kvasir-seg/blob/main/Testing.ipynb) or [script](https://github.com/GerasimovIV/kvasir-seg/blob/main/test.py) as below:

```sh
python test.py
```
