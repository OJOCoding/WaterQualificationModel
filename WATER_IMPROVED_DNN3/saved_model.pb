�
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
SGD/m/cdnn/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_3/bias

+SGD/m/cdnn/dense_3/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_3/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_3/kernel
�
-SGD/m/cdnn/dense_3/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_3/kernel*
_output_shapes

:*
dtype0
�
%SGD/m/cdnn/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/m/cdnn/batch_normalization_2/beta
�
9SGD/m/cdnn/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp%SGD/m/cdnn/batch_normalization_2/beta*
_output_shapes
:*
dtype0
�
&SGD/m/cdnn/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/m/cdnn/batch_normalization_2/gamma
�
:SGD/m/cdnn/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp&SGD/m/cdnn/batch_normalization_2/gamma*
_output_shapes
:*
dtype0
�
%SGD/m/cdnn/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/m/cdnn/batch_normalization_1/beta
�
9SGD/m/cdnn/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp%SGD/m/cdnn/batch_normalization_1/beta*
_output_shapes
:*
dtype0
�
&SGD/m/cdnn/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/m/cdnn/batch_normalization_1/gamma
�
:SGD/m/cdnn/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp&SGD/m/cdnn/batch_normalization_1/gamma*
_output_shapes
:*
dtype0
�
#SGD/m/cdnn/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#SGD/m/cdnn/batch_normalization/beta
�
7SGD/m/cdnn/batch_normalization/beta/Read/ReadVariableOpReadVariableOp#SGD/m/cdnn/batch_normalization/beta*
_output_shapes
:*
dtype0
�
$SGD/m/cdnn/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$SGD/m/cdnn/batch_normalization/gamma
�
8SGD/m/cdnn/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp$SGD/m/cdnn/batch_normalization/gamma*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_2/bias

+SGD/m/cdnn/dense_2/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_2/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_2/kernel
�
-SGD/m/cdnn/dense_2/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_2/kernel*
_output_shapes

:*
dtype0
�
SGD/m/cdnn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_1/bias

+SGD/m/cdnn/dense_1/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_1/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_1/kernel
�
-SGD/m/cdnn/dense_1/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_1/kernel*
_output_shapes

:*
dtype0
�
SGD/m/cdnn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameSGD/m/cdnn/dense/bias
{
)SGD/m/cdnn/dense/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameSGD/m/cdnn/dense/kernel
�
+SGD/m/cdnn/dense/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense/kernel*
_output_shapes

:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
z
cdnn/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_3/bias
s
%cdnn/dense_3/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_3/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_3/kernel
{
'cdnn/dense_3/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_3/kernel*
_output_shapes

:*
dtype0
�
*cdnn/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*cdnn/batch_normalization_2/moving_variance
�
>cdnn/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp*cdnn/batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
�
&cdnn/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&cdnn/batch_normalization_2/moving_mean
�
:cdnn/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp&cdnn/batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
�
*cdnn/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*cdnn/batch_normalization_1/moving_variance
�
>cdnn/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp*cdnn/batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
�
&cdnn/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&cdnn/batch_normalization_1/moving_mean
�
:cdnn/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp&cdnn/batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
�
(cdnn/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(cdnn/batch_normalization/moving_variance
�
<cdnn/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp(cdnn/batch_normalization/moving_variance*
_output_shapes
:*
dtype0
�
$cdnn/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$cdnn/batch_normalization/moving_mean
�
8cdnn/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp$cdnn/batch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
cdnn/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cdnn/batch_normalization_2/beta
�
3cdnn/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization_2/beta*
_output_shapes
:*
dtype0
�
 cdnn/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cdnn/batch_normalization_2/gamma
�
4cdnn/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp cdnn/batch_normalization_2/gamma*
_output_shapes
:*
dtype0
�
cdnn/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cdnn/batch_normalization_1/beta
�
3cdnn/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization_1/beta*
_output_shapes
:*
dtype0
�
 cdnn/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cdnn/batch_normalization_1/gamma
�
4cdnn/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp cdnn/batch_normalization_1/gamma*
_output_shapes
:*
dtype0
�
cdnn/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namecdnn/batch_normalization/beta
�
1cdnn/batch_normalization/beta/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization/beta*
_output_shapes
:*
dtype0
�
cdnn/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name cdnn/batch_normalization/gamma
�
2cdnn/batch_normalization/gamma/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization/gamma*
_output_shapes
:*
dtype0
z
cdnn/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_2/bias
s
%cdnn/dense_2/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_2/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_2/kernel
{
'cdnn/dense_2/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_2/kernel*
_output_shapes

:*
dtype0
z
cdnn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_1/bias
s
%cdnn/dense_1/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_1/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_1/kernel
{
'cdnn/dense_1/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_1/kernel*
_output_shapes

:*
dtype0
v
cdnn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namecdnn/dense/bias
o
#cdnn/dense/bias/Read/ReadVariableOpReadVariableOpcdnn/dense/bias*
_output_shapes
:*
dtype0
~
cdnn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namecdnn/dense/kernel
w
%cdnn/dense/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense/kernel*
_output_shapes

:*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cdnn/dense/kernelcdnn/dense/bias$cdnn/batch_normalization/moving_mean(cdnn/batch_normalization/moving_variancecdnn/batch_normalization/betacdnn/batch_normalization/gammacdnn/dense_1/kernelcdnn/dense_1/bias&cdnn/batch_normalization_1/moving_mean*cdnn/batch_normalization_1/moving_variancecdnn/batch_normalization_1/beta cdnn/batch_normalization_1/gammacdnn/dense_2/kernelcdnn/dense_2/bias&cdnn/batch_normalization_2/moving_mean*cdnn/batch_normalization_2/moving_variancecdnn/batch_normalization_2/beta cdnn/batch_normalization_2/gammacdnn/dense_3/kernelcdnn/dense_3/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_376305

NoOpNoOp
�Y
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�X
value�XB�X B�X
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

Config
	LayerNeurons

HiddenLayers
NormalizationLayers
DropOutLayer
OutputLayer

Classifier
	optimizer

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19*
j
0
1
2
3
4
5
6
7
8
9
10
11
#12
$13*

%0
&1
'2
(3* 
�
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
.trace_0
/trace_1
0trace_2
1trace_3* 
6
2trace_0
3trace_1
4trace_2
5trace_3* 
* 

6DNN.LayerNeurons* 
* 

70
81
92*

:0
;1
<2*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
C_random_generator* 
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

#kernel
$bias*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
o
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T	momentums
U_update_step_xla*

Vserving_default* 
QK
VARIABLE_VALUEcdnn/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEcdnn/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcdnn/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEcdnn/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcdnn/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEcdnn/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcdnn/batch_normalization/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEcdnn/batch_normalization/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE cdnn/batch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEcdnn/batch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_2/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_2/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$cdnn/batch_normalization/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(cdnn/batch_normalization/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_1/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_1/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcdnn/dense_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEcdnn/dense_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*

Wtrace_0* 

Xtrace_0* 

Ytrace_0* 

Ztrace_0* 
.
0
1
2
 3
!4
"5*
C
70
81
92
:3
;4
<5
6
7
8*

[0
\1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

kernel
bias*
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

kernel
bias*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

kernel
bias*
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uaxis
	gamma
beta
moving_mean
moving_variance*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|axis
	gamma
beta
moving_mean
 moving_variance*
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
!moving_mean
"moving_variance*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

#0
$1*

#0
$1*
	
(0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
Q0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*

0
1*

0
1*
	
%0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
	
&0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
	
'0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
 
0
1
2
3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
0
1
2
 3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
0
1
!2
"3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
(0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
b\
VARIABLE_VALUESGD/m/cdnn/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUESGD/m/cdnn/dense/bias1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUESGD/m/cdnn/dense_1/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUESGD/m/cdnn/dense_1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUESGD/m/cdnn/dense_2/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUESGD/m/cdnn/dense_2/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$SGD/m/cdnn/batch_normalization/gamma1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#SGD/m/cdnn/batch_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_1/gamma1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_1/beta2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_2/gamma2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_2/beta2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUESGD/m/cdnn/dense_3/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUESGD/m/cdnn/dense_3/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
	
%0* 
* 
* 
* 
* 
* 
* 
	
&0* 
* 
* 
* 
* 
* 
* 
	
'0* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

0
 1*
* 
* 
* 
* 
* 
* 
* 
* 

!0
"1*
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%cdnn/dense/kernel/Read/ReadVariableOp#cdnn/dense/bias/Read/ReadVariableOp'cdnn/dense_1/kernel/Read/ReadVariableOp%cdnn/dense_1/bias/Read/ReadVariableOp'cdnn/dense_2/kernel/Read/ReadVariableOp%cdnn/dense_2/bias/Read/ReadVariableOp2cdnn/batch_normalization/gamma/Read/ReadVariableOp1cdnn/batch_normalization/beta/Read/ReadVariableOp4cdnn/batch_normalization_1/gamma/Read/ReadVariableOp3cdnn/batch_normalization_1/beta/Read/ReadVariableOp4cdnn/batch_normalization_2/gamma/Read/ReadVariableOp3cdnn/batch_normalization_2/beta/Read/ReadVariableOp8cdnn/batch_normalization/moving_mean/Read/ReadVariableOp<cdnn/batch_normalization/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_1/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_1/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_2/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_2/moving_variance/Read/ReadVariableOp'cdnn/dense_3/kernel/Read/ReadVariableOp%cdnn/dense_3/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp+SGD/m/cdnn/dense/kernel/Read/ReadVariableOp)SGD/m/cdnn/dense/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_1/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_1/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_2/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_2/bias/Read/ReadVariableOp8SGD/m/cdnn/batch_normalization/gamma/Read/ReadVariableOp7SGD/m/cdnn/batch_normalization/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_1/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_1/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_2/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_2/beta/Read/ReadVariableOp-SGD/m/cdnn/dense_3/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_3/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*5
Tin.
,2*	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_377205
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecdnn/dense/kernelcdnn/dense/biascdnn/dense_1/kernelcdnn/dense_1/biascdnn/dense_2/kernelcdnn/dense_2/biascdnn/batch_normalization/gammacdnn/batch_normalization/beta cdnn/batch_normalization_1/gammacdnn/batch_normalization_1/beta cdnn/batch_normalization_2/gammacdnn/batch_normalization_2/beta$cdnn/batch_normalization/moving_mean(cdnn/batch_normalization/moving_variance&cdnn/batch_normalization_1/moving_mean*cdnn/batch_normalization_1/moving_variance&cdnn/batch_normalization_2/moving_mean*cdnn/batch_normalization_2/moving_variancecdnn/dense_3/kernelcdnn/dense_3/bias	iterationlearning_rateSGD/m/cdnn/dense/kernelSGD/m/cdnn/dense/biasSGD/m/cdnn/dense_1/kernelSGD/m/cdnn/dense_1/biasSGD/m/cdnn/dense_2/kernelSGD/m/cdnn/dense_2/bias$SGD/m/cdnn/batch_normalization/gamma#SGD/m/cdnn/batch_normalization/beta&SGD/m/cdnn/batch_normalization_1/gamma%SGD/m/cdnn/batch_normalization_1/beta&SGD/m/cdnn/batch_normalization_2/gamma%SGD/m/cdnn/batch_normalization_2/betaSGD/m/cdnn/dense_3/kernelSGD/m/cdnn/dense_3/biastotal_1count_1totalcount*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_377335��
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_376798

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_1/kernel/Regularizer/L2LossL2Loss=cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_3_376678P
>cdnn_dense_3_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_3_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_3/kernel/Regularizer/L2LossL2Loss=cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'cdnn/dense_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp
�Q
�
__inference__traced_save_377205
file_prefix0
,savev2_cdnn_dense_kernel_read_readvariableop.
*savev2_cdnn_dense_bias_read_readvariableop2
.savev2_cdnn_dense_1_kernel_read_readvariableop0
,savev2_cdnn_dense_1_bias_read_readvariableop2
.savev2_cdnn_dense_2_kernel_read_readvariableop0
,savev2_cdnn_dense_2_bias_read_readvariableop=
9savev2_cdnn_batch_normalization_gamma_read_readvariableop<
8savev2_cdnn_batch_normalization_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_1_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_1_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_2_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_2_beta_read_readvariableopC
?savev2_cdnn_batch_normalization_moving_mean_read_readvariableopG
Csavev2_cdnn_batch_normalization_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_1_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_1_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_2_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_2_moving_variance_read_readvariableop2
.savev2_cdnn_dense_3_kernel_read_readvariableop0
,savev2_cdnn_dense_3_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop6
2savev2_sgd_m_cdnn_dense_kernel_read_readvariableop4
0savev2_sgd_m_cdnn_dense_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_1_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_1_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_2_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_2_bias_read_readvariableopC
?savev2_sgd_m_cdnn_batch_normalization_gamma_read_readvariableopB
>savev2_sgd_m_cdnn_batch_normalization_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_1_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_1_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_2_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_2_beta_read_readvariableop8
4savev2_sgd_m_cdnn_dense_3_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_3_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*�
value�B�)B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_cdnn_dense_kernel_read_readvariableop*savev2_cdnn_dense_bias_read_readvariableop.savev2_cdnn_dense_1_kernel_read_readvariableop,savev2_cdnn_dense_1_bias_read_readvariableop.savev2_cdnn_dense_2_kernel_read_readvariableop,savev2_cdnn_dense_2_bias_read_readvariableop9savev2_cdnn_batch_normalization_gamma_read_readvariableop8savev2_cdnn_batch_normalization_beta_read_readvariableop;savev2_cdnn_batch_normalization_1_gamma_read_readvariableop:savev2_cdnn_batch_normalization_1_beta_read_readvariableop;savev2_cdnn_batch_normalization_2_gamma_read_readvariableop:savev2_cdnn_batch_normalization_2_beta_read_readvariableop?savev2_cdnn_batch_normalization_moving_mean_read_readvariableopCsavev2_cdnn_batch_normalization_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_1_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_1_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_2_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_2_moving_variance_read_readvariableop.savev2_cdnn_dense_3_kernel_read_readvariableop,savev2_cdnn_dense_3_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop2savev2_sgd_m_cdnn_dense_kernel_read_readvariableop0savev2_sgd_m_cdnn_dense_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_1_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_1_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_2_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_2_bias_read_readvariableop?savev2_sgd_m_cdnn_batch_normalization_gamma_read_readvariableop>savev2_sgd_m_cdnn_batch_normalization_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_1_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_1_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_2_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_2_beta_read_readvariableop4savev2_sgd_m_cdnn_dense_3_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_3_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *7
dtypes-
+2)	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::::::::::::::: : ::::::::::::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: 
�$
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_375545

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_376305
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_375392o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�$
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_375463

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
"__inference__traced_restore_377335
file_prefix4
"assignvariableop_cdnn_dense_kernel:0
"assignvariableop_1_cdnn_dense_bias:8
&assignvariableop_2_cdnn_dense_1_kernel:2
$assignvariableop_3_cdnn_dense_1_bias:8
&assignvariableop_4_cdnn_dense_2_kernel:2
$assignvariableop_5_cdnn_dense_2_bias:?
1assignvariableop_6_cdnn_batch_normalization_gamma:>
0assignvariableop_7_cdnn_batch_normalization_beta:A
3assignvariableop_8_cdnn_batch_normalization_1_gamma:@
2assignvariableop_9_cdnn_batch_normalization_1_beta:B
4assignvariableop_10_cdnn_batch_normalization_2_gamma:A
3assignvariableop_11_cdnn_batch_normalization_2_beta:F
8assignvariableop_12_cdnn_batch_normalization_moving_mean:J
<assignvariableop_13_cdnn_batch_normalization_moving_variance:H
:assignvariableop_14_cdnn_batch_normalization_1_moving_mean:L
>assignvariableop_15_cdnn_batch_normalization_1_moving_variance:H
:assignvariableop_16_cdnn_batch_normalization_2_moving_mean:L
>assignvariableop_17_cdnn_batch_normalization_2_moving_variance:9
'assignvariableop_18_cdnn_dense_3_kernel:3
%assignvariableop_19_cdnn_dense_3_bias:'
assignvariableop_20_iteration:	 +
!assignvariableop_21_learning_rate: =
+assignvariableop_22_sgd_m_cdnn_dense_kernel:7
)assignvariableop_23_sgd_m_cdnn_dense_bias:?
-assignvariableop_24_sgd_m_cdnn_dense_1_kernel:9
+assignvariableop_25_sgd_m_cdnn_dense_1_bias:?
-assignvariableop_26_sgd_m_cdnn_dense_2_kernel:9
+assignvariableop_27_sgd_m_cdnn_dense_2_bias:F
8assignvariableop_28_sgd_m_cdnn_batch_normalization_gamma:E
7assignvariableop_29_sgd_m_cdnn_batch_normalization_beta:H
:assignvariableop_30_sgd_m_cdnn_batch_normalization_1_gamma:G
9assignvariableop_31_sgd_m_cdnn_batch_normalization_1_beta:H
:assignvariableop_32_sgd_m_cdnn_batch_normalization_2_gamma:G
9assignvariableop_33_sgd_m_cdnn_batch_normalization_2_beta:?
-assignvariableop_34_sgd_m_cdnn_dense_3_kernel:9
+assignvariableop_35_sgd_m_cdnn_dense_3_bias:%
assignvariableop_36_total_1: %
assignvariableop_37_count_1: #
assignvariableop_38_total: #
assignvariableop_39_count: 
identity_41��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*�
value�B�)B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp"assignvariableop_cdnn_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_cdnn_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_cdnn_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp$assignvariableop_3_cdnn_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_cdnn_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_cdnn_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp1assignvariableop_6_cdnn_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp0assignvariableop_7_cdnn_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp3assignvariableop_8_cdnn_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp2assignvariableop_9_cdnn_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp4assignvariableop_10_cdnn_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp3assignvariableop_11_cdnn_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp8assignvariableop_12_cdnn_batch_normalization_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp<assignvariableop_13_cdnn_batch_normalization_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp:assignvariableop_14_cdnn_batch_normalization_1_moving_meanIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp>assignvariableop_15_cdnn_batch_normalization_1_moving_varianceIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp:assignvariableop_16_cdnn_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp>assignvariableop_17_cdnn_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_cdnn_dense_3_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp%assignvariableop_19_cdnn_dense_3_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_iterationIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_sgd_m_cdnn_dense_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_sgd_m_cdnn_dense_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp-assignvariableop_24_sgd_m_cdnn_dense_1_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_sgd_m_cdnn_dense_1_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp-assignvariableop_26_sgd_m_cdnn_dense_2_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_sgd_m_cdnn_dense_2_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp8assignvariableop_28_sgd_m_cdnn_batch_normalization_gammaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp7assignvariableop_29_sgd_m_cdnn_batch_normalization_betaIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp:assignvariableop_30_sgd_m_cdnn_batch_normalization_1_gammaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp9assignvariableop_31_sgd_m_cdnn_batch_normalization_1_betaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp:assignvariableop_32_sgd_m_cdnn_batch_normalization_2_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp9assignvariableop_33_sgd_m_cdnn_batch_normalization_2_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp-assignvariableop_34_sgd_m_cdnn_dense_3_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp+assignvariableop_35_sgd_m_cdnn_dense_3_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_total_1Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_count_1Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_totalIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_countIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_377028

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_376948

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_376783

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_375690o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_377062

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_375498

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_activation_layer_call_and_return_conditional_losses_376750

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:���������S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_376995

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_375580o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_376807

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_375720o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_375720

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_2/kernel/Regularizer/L2LossL2Loss=cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_376717

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed�[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_376822

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_2/kernel/Regularizer/L2LossL2Loss=cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�

@__inference_cdnn_layer_call_and_return_conditional_losses_376240
input_1
dense_376174:
dense_376176:(
batch_normalization_376179:(
batch_normalization_376181:(
batch_normalization_376183:(
batch_normalization_376185: 
dense_1_376188:
dense_1_376190:*
batch_normalization_1_376193:*
batch_normalization_1_376195:*
batch_normalization_1_376197:*
batch_normalization_1_376199: 
dense_2_376202:
dense_2_376204:*
batch_normalization_2_376207:*
batch_normalization_2_376209:*
batch_normalization_2_376211:*
batch_normalization_2_376213: 
dense_3_376217:
dense_3_376219:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_376174dense_376176*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_375660�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_376179batch_normalization_376181batch_normalization_376183batch_normalization_376185*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_375463�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_376188dense_1_376190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_375690�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_376193batch_normalization_1_376195batch_normalization_1_376197batch_normalization_1_376199*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_375545�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_376202dense_2_376204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_375720�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_376207batch_normalization_2_376209batch_normalization_2_376211batch_normalization_2_376213*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_375627�
dropout/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_375865�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_3_376217dense_3_376219*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_375756�
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_375767�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_376174*
_output_shapes

:*
dtype0�
$cdnn/dense/kernel/Regularizer/L2LossL2Loss;cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#cdnn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_376188*
_output_shapes

:*
dtype0�
&cdnn/dense_1/kernel/Regularizer/L2LossL2Loss=cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_376202*
_output_shapes

:*
dtype0�
&cdnn/dense_2/kernel/Regularizer/L2LossL2Loss=cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_376217*
_output_shapes

:*
dtype0�
&cdnn/dense_3/kernel/Regularizer/L2LossL2Loss=cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
%__inference_cdnn_layer_call_fn_376395
p_tinput
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallp_tinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cdnn_layer_call_and_return_conditional_losses_376014o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_375416

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_375865

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed�[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_layer_call_fn_376835

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_375416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_dense_layer_call_fn_376759

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_375660o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_1_376660P
>cdnn_dense_1_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_1/kernel/Regularizer/L2LossL2Loss=cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'cdnn/dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
A__inference_dense_layer_call_and_return_conditional_losses_376774

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
$cdnn/dense/kernel/Regularizer/L2LossL2Loss;cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#cdnn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_375756

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_3/kernel/Regularizer/L2LossL2Loss=cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_375740

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_375690

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_1/kernel/Regularizer/L2LossL2Loss=cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_376868

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�q
�
!__inference__wrapped_model_375392
input_1;
)cdnn_dense_matmul_readvariableop_resource:8
*cdnn_dense_biasadd_readvariableop_resource:C
5cdnn_batch_normalization_cast_readvariableop_resource:E
7cdnn_batch_normalization_cast_1_readvariableop_resource:E
7cdnn_batch_normalization_cast_2_readvariableop_resource:E
7cdnn_batch_normalization_cast_3_readvariableop_resource:=
+cdnn_dense_1_matmul_readvariableop_resource::
,cdnn_dense_1_biasadd_readvariableop_resource:E
7cdnn_batch_normalization_1_cast_readvariableop_resource:G
9cdnn_batch_normalization_1_cast_1_readvariableop_resource:G
9cdnn_batch_normalization_1_cast_2_readvariableop_resource:G
9cdnn_batch_normalization_1_cast_3_readvariableop_resource:=
+cdnn_dense_2_matmul_readvariableop_resource::
,cdnn_dense_2_biasadd_readvariableop_resource:E
7cdnn_batch_normalization_2_cast_readvariableop_resource:G
9cdnn_batch_normalization_2_cast_1_readvariableop_resource:G
9cdnn_batch_normalization_2_cast_2_readvariableop_resource:G
9cdnn_batch_normalization_2_cast_3_readvariableop_resource:=
+cdnn_dense_3_matmul_readvariableop_resource::
,cdnn_dense_3_biasadd_readvariableop_resource:
identity��,cdnn/batch_normalization/Cast/ReadVariableOp�.cdnn/batch_normalization/Cast_1/ReadVariableOp�.cdnn/batch_normalization/Cast_2/ReadVariableOp�.cdnn/batch_normalization/Cast_3/ReadVariableOp�.cdnn/batch_normalization_1/Cast/ReadVariableOp�0cdnn/batch_normalization_1/Cast_1/ReadVariableOp�0cdnn/batch_normalization_1/Cast_2/ReadVariableOp�0cdnn/batch_normalization_1/Cast_3/ReadVariableOp�.cdnn/batch_normalization_2/Cast/ReadVariableOp�0cdnn/batch_normalization_2/Cast_1/ReadVariableOp�0cdnn/batch_normalization_2/Cast_2/ReadVariableOp�0cdnn/batch_normalization_2/Cast_3/ReadVariableOp�!cdnn/dense/BiasAdd/ReadVariableOp� cdnn/dense/MatMul/ReadVariableOp�#cdnn/dense_1/BiasAdd/ReadVariableOp�"cdnn/dense_1/MatMul/ReadVariableOp�#cdnn/dense_2/BiasAdd/ReadVariableOp�"cdnn/dense_2/MatMul/ReadVariableOp�#cdnn/dense_3/BiasAdd/ReadVariableOp�"cdnn/dense_3/MatMul/ReadVariableOp�
 cdnn/dense/MatMul/ReadVariableOpReadVariableOp)cdnn_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense/MatMulMatMulinput_1(cdnn/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!cdnn/dense/BiasAdd/ReadVariableOpReadVariableOp*cdnn_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense/BiasAddBiasAddcdnn/dense/MatMul:product:0)cdnn/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
cdnn/dense/ReluRelucdnn/dense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,cdnn/batch_normalization/Cast/ReadVariableOpReadVariableOp5cdnn_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
.cdnn/batch_normalization/Cast_1/ReadVariableOpReadVariableOp7cdnn_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
.cdnn/batch_normalization/Cast_2/ReadVariableOpReadVariableOp7cdnn_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
.cdnn/batch_normalization/Cast_3/ReadVariableOpReadVariableOp7cdnn_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0m
(cdnn/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&cdnn/batch_normalization/batchnorm/addAddV26cdnn/batch_normalization/Cast_1/ReadVariableOp:value:01cdnn/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization/batchnorm/RsqrtRsqrt*cdnn/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
&cdnn/batch_normalization/batchnorm/mulMul,cdnn/batch_normalization/batchnorm/Rsqrt:y:06cdnn/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization/batchnorm/mul_1Mulcdnn/dense/Relu:activations:0*cdnn/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
(cdnn/batch_normalization/batchnorm/mul_2Mul4cdnn/batch_normalization/Cast/ReadVariableOp:value:0*cdnn/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
&cdnn/batch_normalization/batchnorm/subSub6cdnn/batch_normalization/Cast_2/ReadVariableOp:value:0,cdnn/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization/batchnorm/add_1AddV2,cdnn/batch_normalization/batchnorm/mul_1:z:0*cdnn/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
"cdnn/dense_1/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_1/MatMulMatMul,cdnn/batch_normalization/batchnorm/add_1:z:0*cdnn/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_1/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_1/BiasAddBiasAddcdnn/dense_1/MatMul:product:0+cdnn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
cdnn/dense_1/ReluRelucdnn/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.cdnn/batch_normalization_1/Cast/ReadVariableOpReadVariableOp7cdnn_batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp9cdnn_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp9cdnn_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp9cdnn_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0o
*cdnn/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(cdnn/batch_normalization_1/batchnorm/addAddV28cdnn/batch_normalization_1/Cast_1/ReadVariableOp:value:03cdnn/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_1/batchnorm/RsqrtRsqrt,cdnn/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_1/batchnorm/mulMul.cdnn/batch_normalization_1/batchnorm/Rsqrt:y:08cdnn/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_1/batchnorm/mul_1Mulcdnn/dense_1/Relu:activations:0,cdnn/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
*cdnn/batch_normalization_1/batchnorm/mul_2Mul6cdnn/batch_normalization_1/Cast/ReadVariableOp:value:0,cdnn/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_1/batchnorm/subSub8cdnn/batch_normalization_1/Cast_2/ReadVariableOp:value:0.cdnn/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_1/batchnorm/add_1AddV2.cdnn/batch_normalization_1/batchnorm/mul_1:z:0,cdnn/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
"cdnn/dense_2/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_2/MatMulMatMul.cdnn/batch_normalization_1/batchnorm/add_1:z:0*cdnn/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_2/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_2/BiasAddBiasAddcdnn/dense_2/MatMul:product:0+cdnn/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
cdnn/dense_2/ReluRelucdnn/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.cdnn/batch_normalization_2/Cast/ReadVariableOpReadVariableOp7cdnn_batch_normalization_2_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp9cdnn_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp9cdnn_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp9cdnn_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0o
*cdnn/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(cdnn/batch_normalization_2/batchnorm/addAddV28cdnn/batch_normalization_2/Cast_1/ReadVariableOp:value:03cdnn/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_2/batchnorm/RsqrtRsqrt,cdnn/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_2/batchnorm/mulMul.cdnn/batch_normalization_2/batchnorm/Rsqrt:y:08cdnn/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_2/batchnorm/mul_1Mulcdnn/dense_2/Relu:activations:0,cdnn/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
*cdnn/batch_normalization_2/batchnorm/mul_2Mul6cdnn/batch_normalization_2/Cast/ReadVariableOp:value:0,cdnn/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_2/batchnorm/subSub8cdnn/batch_normalization_2/Cast_2/ReadVariableOp:value:0.cdnn/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_2/batchnorm/add_1AddV2.cdnn/batch_normalization_2/batchnorm/mul_1:z:0,cdnn/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
cdnn/dropout/IdentityIdentity.cdnn/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:����������
"cdnn/dense_3/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_3/MatMulMatMulcdnn/dropout/Identity:output:0*cdnn/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_3/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_3/BiasAddBiasAddcdnn/dense_3/MatMul:product:0+cdnn/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s
cdnn/activation/SigmoidSigmoidcdnn/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentitycdnn/activation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^cdnn/batch_normalization/Cast/ReadVariableOp/^cdnn/batch_normalization/Cast_1/ReadVariableOp/^cdnn/batch_normalization/Cast_2/ReadVariableOp/^cdnn/batch_normalization/Cast_3/ReadVariableOp/^cdnn/batch_normalization_1/Cast/ReadVariableOp1^cdnn/batch_normalization_1/Cast_1/ReadVariableOp1^cdnn/batch_normalization_1/Cast_2/ReadVariableOp1^cdnn/batch_normalization_1/Cast_3/ReadVariableOp/^cdnn/batch_normalization_2/Cast/ReadVariableOp1^cdnn/batch_normalization_2/Cast_1/ReadVariableOp1^cdnn/batch_normalization_2/Cast_2/ReadVariableOp1^cdnn/batch_normalization_2/Cast_3/ReadVariableOp"^cdnn/dense/BiasAdd/ReadVariableOp!^cdnn/dense/MatMul/ReadVariableOp$^cdnn/dense_1/BiasAdd/ReadVariableOp#^cdnn/dense_1/MatMul/ReadVariableOp$^cdnn/dense_2/BiasAdd/ReadVariableOp#^cdnn/dense_2/MatMul/ReadVariableOp$^cdnn/dense_3/BiasAdd/ReadVariableOp#^cdnn/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 2\
,cdnn/batch_normalization/Cast/ReadVariableOp,cdnn/batch_normalization/Cast/ReadVariableOp2`
.cdnn/batch_normalization/Cast_1/ReadVariableOp.cdnn/batch_normalization/Cast_1/ReadVariableOp2`
.cdnn/batch_normalization/Cast_2/ReadVariableOp.cdnn/batch_normalization/Cast_2/ReadVariableOp2`
.cdnn/batch_normalization/Cast_3/ReadVariableOp.cdnn/batch_normalization/Cast_3/ReadVariableOp2`
.cdnn/batch_normalization_1/Cast/ReadVariableOp.cdnn/batch_normalization_1/Cast/ReadVariableOp2d
0cdnn/batch_normalization_1/Cast_1/ReadVariableOp0cdnn/batch_normalization_1/Cast_1/ReadVariableOp2d
0cdnn/batch_normalization_1/Cast_2/ReadVariableOp0cdnn/batch_normalization_1/Cast_2/ReadVariableOp2d
0cdnn/batch_normalization_1/Cast_3/ReadVariableOp0cdnn/batch_normalization_1/Cast_3/ReadVariableOp2`
.cdnn/batch_normalization_2/Cast/ReadVariableOp.cdnn/batch_normalization_2/Cast/ReadVariableOp2d
0cdnn/batch_normalization_2/Cast_1/ReadVariableOp0cdnn/batch_normalization_2/Cast_1/ReadVariableOp2d
0cdnn/batch_normalization_2/Cast_2/ReadVariableOp0cdnn/batch_normalization_2/Cast_2/ReadVariableOp2d
0cdnn/batch_normalization_2/Cast_3/ReadVariableOp0cdnn/batch_normalization_2/Cast_3/ReadVariableOp2F
!cdnn/dense/BiasAdd/ReadVariableOp!cdnn/dense/BiasAdd/ReadVariableOp2D
 cdnn/dense/MatMul/ReadVariableOp cdnn/dense/MatMul/ReadVariableOp2J
#cdnn/dense_1/BiasAdd/ReadVariableOp#cdnn/dense_1/BiasAdd/ReadVariableOp2H
"cdnn/dense_1/MatMul/ReadVariableOp"cdnn/dense_1/MatMul/ReadVariableOp2J
#cdnn/dense_2/BiasAdd/ReadVariableOp#cdnn/dense_2/BiasAdd/ReadVariableOp2H
"cdnn/dense_2/MatMul/ReadVariableOp"cdnn/dense_2/MatMul/ReadVariableOp2J
#cdnn/dense_3/BiasAdd/ReadVariableOp#cdnn/dense_3/BiasAdd/ReadVariableOp2H
"cdnn/dense_3/MatMul/ReadVariableOp"cdnn/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
��
�
@__inference_cdnn_layer_call_and_return_conditional_losses_376492
p_tinput6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:>
0batch_normalization_cast_readvariableop_resource:@
2batch_normalization_cast_1_readvariableop_resource:@
2batch_normalization_cast_2_readvariableop_resource:@
2batch_normalization_cast_3_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:@
2batch_normalization_1_cast_readvariableop_resource:B
4batch_normalization_1_cast_1_readvariableop_resource:B
4batch_normalization_1_cast_2_readvariableop_resource:B
4batch_normalization_1_cast_3_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:@
2batch_normalization_2_cast_readvariableop_resource:B
4batch_normalization_2_cast_1_readvariableop_resource:B
4batch_normalization_2_cast_2_readvariableop_resource:B
4batch_normalization_2_cast_3_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity��'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�)batch_normalization/Cast_2/ReadVariableOp�)batch_normalization/Cast_3/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�+batch_normalization_1/Cast_2/ReadVariableOp�+batch_normalization_1/Cast_3/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�+batch_normalization_2/Cast_2/ReadVariableOp�+batch_normalization_2/Cast_3/ReadVariableOp�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0w
dense/MatMulMatMulp_tinput#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/mul_1Muldense/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_1/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_1/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_2/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_2/batchnorm/mul_1Muldense_2/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������y
dropout/IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_3/MatMulMatMuldropout/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
activation/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:����������
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
$cdnn/dense/kernel/Regularizer/L2LossL2Loss;cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#cdnn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_1/kernel/Regularizer/L2LossL2Loss=cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_2/kernel/Regularizer/L2LossL2Loss=cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_3/kernel/Regularizer/L2LossL2Loss=cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2Z
+batch_normalization_2/Cast_2/ReadVariableOp+batch_normalization_2/Cast_2/ReadVariableOp2Z
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�I
�

@__inference_cdnn_layer_call_and_return_conditional_losses_376171
input_1
dense_376105:
dense_376107:(
batch_normalization_376110:(
batch_normalization_376112:(
batch_normalization_376114:(
batch_normalization_376116: 
dense_1_376119:
dense_1_376121:*
batch_normalization_1_376124:*
batch_normalization_1_376126:*
batch_normalization_1_376128:*
batch_normalization_1_376130: 
dense_2_376133:
dense_2_376135:*
batch_normalization_2_376138:*
batch_normalization_2_376140:*
batch_normalization_2_376142:*
batch_normalization_2_376144: 
dense_3_376148:
dense_3_376150:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_376105dense_376107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_375660�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_376110batch_normalization_376112batch_normalization_376114batch_normalization_376116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_375416�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_376119dense_1_376121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_375690�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_376124batch_normalization_1_376126batch_normalization_1_376128batch_normalization_1_376130*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_375498�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_376133dense_2_376135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_375720�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_376138batch_normalization_2_376140batch_normalization_2_376142batch_normalization_2_376144*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_375580�
dropout/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_375740�
dense_3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_3_376148dense_3_376150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_375756�
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_375767�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_376105*
_output_shapes

:*
dtype0�
$cdnn/dense/kernel/Regularizer/L2LossL2Loss;cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#cdnn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_376119*
_output_shapes

:*
dtype0�
&cdnn/dense_1/kernel/Regularizer/L2LossL2Loss=cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_376133*
_output_shapes

:*
dtype0�
&cdnn/dense_2/kernel/Regularizer/L2LossL2Loss=cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_376148*
_output_shapes

:*
dtype0�
&cdnn/dense_3/kernel/Regularizer/L2LossL2Loss=cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�$
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_376982

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_376740

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_3/kernel/Regularizer/L2LossL2Loss=cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_375580

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_376902

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
(__inference_dropout_layer_call_fn_376700

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_375865o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_376928

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_375545o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_cdnn_layer_call_fn_376350
p_tinput
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallp_tinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cdnn_layer_call_and_return_conditional_losses_375786o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�J
�

@__inference_cdnn_layer_call_and_return_conditional_losses_376014
p_tinput
dense_375948:
dense_375950:(
batch_normalization_375953:(
batch_normalization_375955:(
batch_normalization_375957:(
batch_normalization_375959: 
dense_1_375962:
dense_1_375964:*
batch_normalization_1_375967:*
batch_normalization_1_375969:*
batch_normalization_1_375971:*
batch_normalization_1_375973: 
dense_2_375976:
dense_2_375978:*
batch_normalization_2_375981:*
batch_normalization_2_375983:*
batch_normalization_2_375985:*
batch_normalization_2_375987: 
dense_3_375991:
dense_3_375993:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallp_tinputdense_375948dense_375950*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_375660�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_375953batch_normalization_375955batch_normalization_375957batch_normalization_375959*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_375463�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_375962dense_1_375964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_375690�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_375967batch_normalization_1_375969batch_normalization_1_375971batch_normalization_1_375973*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_375545�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_375976dense_2_375978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_375720�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_375981batch_normalization_2_375983batch_normalization_2_375985batch_normalization_2_375987*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_375627�
dropout/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_375865�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_3_375991dense_3_375993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_375756�
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_375767�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_375948*
_output_shapes

:*
dtype0�
$cdnn/dense/kernel/Regularizer/L2LossL2Loss;cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#cdnn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_375962*
_output_shapes

:*
dtype0�
&cdnn/dense_1/kernel/Regularizer/L2LossL2Loss=cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_375976*
_output_shapes

:*
dtype0�
&cdnn/dense_2/kernel/Regularizer/L2LossL2Loss=cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_375991*
_output_shapes

:*
dtype0�
&cdnn/dense_3/kernel/Regularizer/L2LossL2Loss=cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tInput
�
�
(__inference_dense_3_layer_call_fn_376726

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_375756o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�I
�

@__inference_cdnn_layer_call_and_return_conditional_losses_375786
p_tinput
dense_375661:
dense_375663:(
batch_normalization_375666:(
batch_normalization_375668:(
batch_normalization_375670:(
batch_normalization_375672: 
dense_1_375691:
dense_1_375693:*
batch_normalization_1_375696:*
batch_normalization_1_375698:*
batch_normalization_1_375700:*
batch_normalization_1_375702: 
dense_2_375721:
dense_2_375723:*
batch_normalization_2_375726:*
batch_normalization_2_375728:*
batch_normalization_2_375730:*
batch_normalization_2_375732: 
dense_3_375757:
dense_3_375759:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallp_tinputdense_375661dense_375663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_375660�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_375666batch_normalization_375668batch_normalization_375670batch_normalization_375672*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_375416�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_375691dense_1_375693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_375690�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_375696batch_normalization_1_375698batch_normalization_1_375700batch_normalization_1_375702*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_375498�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_375721dense_2_375723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_375720�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_375726batch_normalization_2_375728batch_normalization_2_375730batch_normalization_2_375732*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_375580�
dropout/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_375740�
dense_3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_3_375757dense_3_375759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_375756�
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_375767�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_375661*
_output_shapes

:*
dtype0�
$cdnn/dense/kernel/Regularizer/L2LossL2Loss;cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#cdnn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_375691*
_output_shapes

:*
dtype0�
&cdnn/dense_1/kernel/Regularizer/L2LossL2Loss=cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_375721*
_output_shapes

:*
dtype0�
&cdnn/dense_2/kernel/Regularizer/L2LossL2Loss=cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_375757*
_output_shapes

:*
dtype0�
&cdnn/dense_3/kernel/Regularizer/L2LossL2Loss=cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tInput
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_376705

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_375660

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
$cdnn/dense/kernel/Regularizer/L2LossL2Loss;cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#cdnn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_377008

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_375627o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_layer_call_fn_376848

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_375463o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_cdnn_layer_call_fn_375829
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cdnn_layer_call_and_return_conditional_losses_375786o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
b
F__inference_activation_layer_call_and_return_conditional_losses_375767

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:���������S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_cdnn_layer_call_fn_376102
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cdnn_layer_call_and_return_conditional_losses_376014o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
��
�
@__inference_cdnn_layer_call_and_return_conditional_losses_376638
p_tinput6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:I
;batch_normalization_assignmovingavg_readvariableop_resource:K
=batch_normalization_assignmovingavg_1_readvariableop_resource:>
0batch_normalization_cast_readvariableop_resource:@
2batch_normalization_cast_1_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:K
=batch_normalization_1_assignmovingavg_readvariableop_resource:M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_1_cast_readvariableop_resource:B
4batch_normalization_1_cast_1_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:K
=batch_normalization_2_assignmovingavg_readvariableop_resource:M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_2_cast_readvariableop_resource:B
4batch_normalization_2_cast_1_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0w
dense/MatMulMatMulp_tinput#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
 batch_normalization/moments/meanMeandense/Relu:activations:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:�
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense/Relu:activations:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/mul_1Muldense/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_1/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_1/moments/meanMeandense_1/Relu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/Relu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_1/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_2/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_2/moments/meanMeandense_2/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_2/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_2/batchnorm/mul_1Muldense_2/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout/dropout/MulMul)batch_normalization_2/batchnorm/add_1:z:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������n
dropout/dropout/ShapeShape)batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed�c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*'
_output_shapes
:����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_3/MatMulMatMul!dropout/dropout/SelectV2:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
activation/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:����������
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
$cdnn/dense/kernel/Regularizer/L2LossL2Loss;cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#cdnn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_1/kernel/Regularizer/L2LossL2Loss=cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_2/kernel/Regularizer/L2LossL2Loss=cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_3/kernel/Regularizer/L2LossL2Loss=cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�	
�
__inference_loss_fn_0_376651N
<cdnn_dense_kernel_regularizer_l2loss_readvariableop_resource:
identity��3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<cdnn_dense_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
$cdnn/dense/kernel/Regularizer/L2LossL2Loss;cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#cdnn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c
IdentityIdentity%cdnn/dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp
�
G
+__inference_activation_layer_call_fn_376745

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_375767`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_dropout_layer_call_fn_376695

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_375740`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_376915

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_375498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_2_376669P
>cdnn_dense_2_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_2_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_2/kernel/Regularizer/L2LossL2Loss=cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'cdnn/dense_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp
�$
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_375627

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:ʕ
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

Config
	LayerNeurons

HiddenLayers
NormalizationLayers
DropOutLayer
OutputLayer

Classifier
	optimizer

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
#12
$13"
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
�
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
.trace_0
/trace_1
0trace_2
1trace_32�
%__inference_cdnn_layer_call_fn_375829
%__inference_cdnn_layer_call_fn_376350
%__inference_cdnn_layer_call_fn_376395
%__inference_cdnn_layer_call_fn_376102�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z.trace_0z/trace_1z0trace_2z1trace_3
�
2trace_0
3trace_1
4trace_2
5trace_32�
@__inference_cdnn_layer_call_and_return_conditional_losses_376492
@__inference_cdnn_layer_call_and_return_conditional_losses_376638
@__inference_cdnn_layer_call_and_return_conditional_losses_376171
@__inference_cdnn_layer_call_and_return_conditional_losses_376240�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z2trace_0z3trace_1z4trace_2z5trace_3
�B�
!__inference__wrapped_model_375392input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
6
6DNN.LayerNeurons"
trackable_dict_wrapper
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
C_random_generator"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
�
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T	momentums
U_update_step_xla"
experimentalOptimizer
,
Vserving_default"
signature_map
#:!2cdnn/dense/kernel
:2cdnn/dense/bias
%:#2cdnn/dense_1/kernel
:2cdnn/dense_1/bias
%:#2cdnn/dense_2/kernel
:2cdnn/dense_2/bias
,:*2cdnn/batch_normalization/gamma
+:)2cdnn/batch_normalization/beta
.:,2 cdnn/batch_normalization_1/gamma
-:+2cdnn/batch_normalization_1/beta
.:,2 cdnn/batch_normalization_2/gamma
-:+2cdnn/batch_normalization_2/beta
4:2 (2$cdnn/batch_normalization/moving_mean
8:6 (2(cdnn/batch_normalization/moving_variance
6:4 (2&cdnn/batch_normalization_1/moving_mean
::8 (2*cdnn/batch_normalization_1/moving_variance
6:4 (2&cdnn/batch_normalization_2/moving_mean
::8 (2*cdnn/batch_normalization_2/moving_variance
%:#2cdnn/dense_3/kernel
:2cdnn/dense_3/bias
�
Wtrace_02�
__inference_loss_fn_0_376651�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zWtrace_0
�
Xtrace_02�
__inference_loss_fn_1_376660�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zXtrace_0
�
Ytrace_02�
__inference_loss_fn_2_376669�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zYtrace_0
�
Ztrace_02�
__inference_loss_fn_3_376678�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zZtrace_0
J
0
1
2
 3
!4
"5"
trackable_list_wrapper
_
70
81
92
:3
;4
<5
6
7
8"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_cdnn_layer_call_fn_375829input_1"�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
%__inference_cdnn_layer_call_fn_376350p_tinput"�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
%__inference_cdnn_layer_call_fn_376395p_tinput"�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
%__inference_cdnn_layer_call_fn_376102input_1"�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
@__inference_cdnn_layer_call_and_return_conditional_losses_376492p_tinput"�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
@__inference_cdnn_layer_call_and_return_conditional_losses_376638p_tinput"�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
@__inference_cdnn_layer_call_and_return_conditional_losses_376171input_1"�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
@__inference_cdnn_layer_call_and_return_conditional_losses_376240input_1"�
���
FullArgSpec
args�
jself

jp_tInput
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
 "
trackable_list_wrapper
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uaxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|axis
	gamma
beta
moving_mean
 moving_variance"
_tf_keras_layer
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
!moving_mean
"moving_variance"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_layer_call_fn_376695
(__inference_dropout_layer_call_fn_376700�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_layer_call_and_return_conditional_losses_376705
C__inference_dropout_layer_call_and_return_conditional_losses_376717�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_3_layer_call_fn_376726�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_376740�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_activation_layer_call_fn_376745�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_activation_layer_call_and_return_conditional_losses_376750�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
Q0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�2��
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_376305input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_376651"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_376660"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_376669"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_376678"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
%0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_376759�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_376774�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_376783�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_376798�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
'0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_376807�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_376822�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_layer_call_fn_376835
4__inference_batch_normalization_layer_call_fn_376848�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_376868
O__inference_batch_normalization_layer_call_and_return_conditional_losses_376902�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_1_layer_call_fn_376915
6__inference_batch_normalization_1_layer_call_fn_376928�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_376948
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_376982�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
0
1
!2
"3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_2_layer_call_fn_376995
6__inference_batch_normalization_2_layer_call_fn_377008�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_377028
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_377062�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dropout_layer_call_fn_376695inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_layer_call_fn_376700inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_376705inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_376717inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_3_layer_call_fn_376726inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_3_layer_call_and_return_conditional_losses_376740inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_activation_layer_call_fn_376745inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_activation_layer_call_and_return_conditional_losses_376750inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
':%2SGD/m/cdnn/dense/kernel
!:2SGD/m/cdnn/dense/bias
):'2SGD/m/cdnn/dense_1/kernel
#:!2SGD/m/cdnn/dense_1/bias
):'2SGD/m/cdnn/dense_2/kernel
#:!2SGD/m/cdnn/dense_2/bias
0:.2$SGD/m/cdnn/batch_normalization/gamma
/:-2#SGD/m/cdnn/batch_normalization/beta
2:02&SGD/m/cdnn/batch_normalization_1/gamma
1:/2%SGD/m/cdnn/batch_normalization_1/beta
2:02&SGD/m/cdnn/batch_normalization_2/gamma
1:/2%SGD/m/cdnn/batch_normalization_2/beta
):'2SGD/m/cdnn/dense_3/kernel
#:!2SGD/m/cdnn/dense_3/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_layer_call_fn_376759inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_376774inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_1_layer_call_fn_376783inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_376798inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_376807inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_376822inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_layer_call_fn_376835inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_layer_call_fn_376848inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_376868inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_376902inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_1_layer_call_fn_376915inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_1_layer_call_fn_376928inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_376948inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_376982inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_2_layer_call_fn_376995inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_2_layer_call_fn_377008inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_377028inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_377062inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_375392} !"#$0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
F__inference_activation_layer_call_and_return_conditional_losses_376750_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
+__inference_activation_layer_call_fn_376745T/�,
%�"
 �
inputs���������
� "!�
unknown����������
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_376948i 3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_376982i 3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
6__inference_batch_normalization_1_layer_call_fn_376915^ 3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
6__inference_batch_normalization_1_layer_call_fn_376928^ 3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_377028i!"3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_377062i!"3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
6__inference_batch_normalization_2_layer_call_fn_376995^!"3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
6__inference_batch_normalization_2_layer_call_fn_377008^!"3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
O__inference_batch_normalization_layer_call_and_return_conditional_losses_376868i3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_376902i3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
4__inference_batch_normalization_layer_call_fn_376835^3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
4__inference_batch_normalization_layer_call_fn_376848^3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
@__inference_cdnn_layer_call_and_return_conditional_losses_376171� !"#$@�=
&�#
!�
input_1���������
�

trainingp ",�)
"�
tensor_0���������
� �
@__inference_cdnn_layer_call_and_return_conditional_losses_376240� !"#$@�=
&�#
!�
input_1���������
�

trainingp",�)
"�
tensor_0���������
� �
@__inference_cdnn_layer_call_and_return_conditional_losses_376492� !"#$A�>
'�$
"�
p_tinput���������
�

trainingp ",�)
"�
tensor_0���������
� �
@__inference_cdnn_layer_call_and_return_conditional_losses_376638� !"#$A�>
'�$
"�
p_tinput���������
�

trainingp",�)
"�
tensor_0���������
� �
%__inference_cdnn_layer_call_fn_375829{ !"#$@�=
&�#
!�
input_1���������
�

trainingp "!�
unknown����������
%__inference_cdnn_layer_call_fn_376102{ !"#$@�=
&�#
!�
input_1���������
�

trainingp"!�
unknown����������
%__inference_cdnn_layer_call_fn_376350| !"#$A�>
'�$
"�
p_tinput���������
�

trainingp "!�
unknown����������
%__inference_cdnn_layer_call_fn_376395| !"#$A�>
'�$
"�
p_tinput���������
�

trainingp"!�
unknown����������
C__inference_dense_1_layer_call_and_return_conditional_losses_376798c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_1_layer_call_fn_376783X/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dense_2_layer_call_and_return_conditional_losses_376822c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_2_layer_call_fn_376807X/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dense_3_layer_call_and_return_conditional_losses_376740c#$/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_3_layer_call_fn_376726X#$/�,
%�"
 �
inputs���������
� "!�
unknown����������
A__inference_dense_layer_call_and_return_conditional_losses_376774c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
&__inference_dense_layer_call_fn_376759X/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dropout_layer_call_and_return_conditional_losses_376705c3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
C__inference_dropout_layer_call_and_return_conditional_losses_376717c3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
(__inference_dropout_layer_call_fn_376695X3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
(__inference_dropout_layer_call_fn_376700X3�0
)�&
 �
inputs���������
p
� "!�
unknown���������D
__inference_loss_fn_0_376651$�

� 
� "�
unknown D
__inference_loss_fn_1_376660$�

� 
� "�
unknown D
__inference_loss_fn_2_376669$�

� 
� "�
unknown D
__inference_loss_fn_3_376678$#�

� 
� "�
unknown �
$__inference_signature_wrapper_376305� !"#$;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������