��
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
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��
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
SGD/m/cdnn/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_6/bias

+SGD/m/cdnn/dense_6/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_6/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_6/kernel
�
-SGD/m/cdnn/dense_6/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_6/kernel*
_output_shapes

:*
dtype0
�
%SGD/m/cdnn/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/m/cdnn/batch_normalization_5/beta
�
9SGD/m/cdnn/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp%SGD/m/cdnn/batch_normalization_5/beta*
_output_shapes
:*
dtype0
�
&SGD/m/cdnn/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/m/cdnn/batch_normalization_5/gamma
�
:SGD/m/cdnn/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp&SGD/m/cdnn/batch_normalization_5/gamma*
_output_shapes
:*
dtype0
�
%SGD/m/cdnn/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/m/cdnn/batch_normalization_4/beta
�
9SGD/m/cdnn/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp%SGD/m/cdnn/batch_normalization_4/beta*
_output_shapes
:*
dtype0
�
&SGD/m/cdnn/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/m/cdnn/batch_normalization_4/gamma
�
:SGD/m/cdnn/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp&SGD/m/cdnn/batch_normalization_4/gamma*
_output_shapes
:*
dtype0
�
%SGD/m/cdnn/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/m/cdnn/batch_normalization_3/beta
�
9SGD/m/cdnn/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp%SGD/m/cdnn/batch_normalization_3/beta*
_output_shapes
:*
dtype0
�
&SGD/m/cdnn/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/m/cdnn/batch_normalization_3/gamma
�
:SGD/m/cdnn/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp&SGD/m/cdnn/batch_normalization_3/gamma*
_output_shapes
:*
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
SGD/m/cdnn/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_5/bias

+SGD/m/cdnn/dense_5/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_5/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_5/kernel
�
-SGD/m/cdnn/dense_5/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_5/kernel*
_output_shapes

:*
dtype0
�
SGD/m/cdnn/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_4/bias

+SGD/m/cdnn/dense_4/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_4/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_4/kernel
�
-SGD/m/cdnn/dense_4/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_4/kernel*
_output_shapes

:*
dtype0
�
SGD/m/cdnn/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_3/bias

+SGD/m/cdnn/dense_3/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_3/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_3/kernel
�
-SGD/m/cdnn/dense_3/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_3/kernel*
_output_shapes

:*
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
cdnn/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_6/bias
s
%cdnn/dense_6/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_6/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_6/kernel
{
'cdnn/dense_6/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_6/kernel*
_output_shapes

:*
dtype0
�
*cdnn/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*cdnn/batch_normalization_5/moving_variance
�
>cdnn/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp*cdnn/batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
�
&cdnn/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&cdnn/batch_normalization_5/moving_mean
�
:cdnn/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp&cdnn/batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
�
*cdnn/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*cdnn/batch_normalization_4/moving_variance
�
>cdnn/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp*cdnn/batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
�
&cdnn/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&cdnn/batch_normalization_4/moving_mean
�
:cdnn/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp&cdnn/batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
�
*cdnn/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*cdnn/batch_normalization_3/moving_variance
�
>cdnn/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp*cdnn/batch_normalization_3/moving_variance*
_output_shapes
:*
dtype0
�
&cdnn/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&cdnn/batch_normalization_3/moving_mean
�
:cdnn/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp&cdnn/batch_normalization_3/moving_mean*
_output_shapes
:*
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
cdnn/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cdnn/batch_normalization_5/beta
�
3cdnn/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization_5/beta*
_output_shapes
:*
dtype0
�
 cdnn/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cdnn/batch_normalization_5/gamma
�
4cdnn/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp cdnn/batch_normalization_5/gamma*
_output_shapes
:*
dtype0
�
cdnn/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cdnn/batch_normalization_4/beta
�
3cdnn/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization_4/beta*
_output_shapes
:*
dtype0
�
 cdnn/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cdnn/batch_normalization_4/gamma
�
4cdnn/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp cdnn/batch_normalization_4/gamma*
_output_shapes
:*
dtype0
�
cdnn/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cdnn/batch_normalization_3/beta
�
3cdnn/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization_3/beta*
_output_shapes
:*
dtype0
�
 cdnn/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cdnn/batch_normalization_3/gamma
�
4cdnn/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp cdnn/batch_normalization_3/gamma*
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
cdnn/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_5/bias
s
%cdnn/dense_5/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_5/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_5/kernel
{
'cdnn/dense_5/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_5/kernel*
_output_shapes

:*
dtype0
z
cdnn/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_4/bias
s
%cdnn/dense_4/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_4/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_4/kernel
{
'cdnn/dense_4/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_4/kernel*
_output_shapes

:*
dtype0
z
cdnn/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_3/bias
s
%cdnn/dense_3/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_3/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_3/kernel
{
'cdnn/dense_3/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_3/kernel*
_output_shapes

:*
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
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cdnn/dense/kernelcdnn/dense/bias$cdnn/batch_normalization/moving_mean(cdnn/batch_normalization/moving_variancecdnn/batch_normalization/betacdnn/batch_normalization/gammacdnn/dense_1/kernelcdnn/dense_1/bias&cdnn/batch_normalization_1/moving_mean*cdnn/batch_normalization_1/moving_variancecdnn/batch_normalization_1/beta cdnn/batch_normalization_1/gammacdnn/dense_2/kernelcdnn/dense_2/bias&cdnn/batch_normalization_2/moving_mean*cdnn/batch_normalization_2/moving_variancecdnn/batch_normalization_2/beta cdnn/batch_normalization_2/gammacdnn/dense_3/kernelcdnn/dense_3/bias&cdnn/batch_normalization_3/moving_mean*cdnn/batch_normalization_3/moving_variancecdnn/batch_normalization_3/beta cdnn/batch_normalization_3/gammacdnn/dense_4/kernelcdnn/dense_4/bias&cdnn/batch_normalization_4/moving_mean*cdnn/batch_normalization_4/moving_variancecdnn/batch_normalization_4/beta cdnn/batch_normalization_4/gammacdnn/dense_5/kernelcdnn/dense_5/bias&cdnn/batch_normalization_5/moving_mean*cdnn/batch_normalization_5/moving_variancecdnn/batch_normalization_5/beta cdnn/batch_normalization_5/gammacdnn/dense_6/kernelcdnn/dense_6/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_181937

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
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
�
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
$19
%20
&21
'22
(23
)24
*25
+26
,27
-28
.29
/30
031
132
233
334
435
536
637*
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
$19
%20
&21
'22
(23
524
625*
3
70
81
92
:3
;4
<5
=6* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
* 

KDNN.LayerNeurons* 
* 
.
L0
M1
N2
O3
P4
Q5*
.
R0
S1
T2
U3
V4
W5*
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator* 
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

5kernel
6bias*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses* 
o
k
_variables
l_iterations
m_learning_rate
n_index_dict
o	momentums
p_update_step_xla*

qserving_default* 
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
SM
VARIABLE_VALUEcdnn/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEcdnn/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcdnn/dense_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEcdnn/dense_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcdnn/dense_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEcdnn/dense_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEcdnn/batch_normalization/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcdnn/batch_normalization/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_1/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_1/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_2/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_2/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_3/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_3/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_4/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_4/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_5/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_5/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$cdnn/batch_normalization/moving_mean'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(cdnn/batch_normalization/moving_variance'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_1/moving_mean'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_1/moving_variance'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_2/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_2/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_3/moving_mean'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_3/moving_variance'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_4/moving_mean'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_4/moving_variance'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_5/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_5/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcdnn/dense_6/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEcdnn/dense_6/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*

rtrace_0* 

strace_0* 

ttrace_0* 

utrace_0* 

vtrace_0* 

wtrace_0* 

xtrace_0* 
Z
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411*
r
L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9
V10
W11
12
13
14*

y0
z1*
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
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
)moving_mean
*moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
 beta
+moving_mean
,moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	!gamma
"beta
-moving_mean
.moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	#gamma
$beta
/moving_mean
0moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	%gamma
&beta
1moving_mean
2moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	'gamma
(beta
3moving_mean
4moving_variance*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

50
61*

50
61*
	
=0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*
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
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
l0
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
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
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
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25*
* 
* 
* 
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

�total

�count
�
_fn_kwargs*

0
1*

0
1*
	
70* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
	
80* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
	
90* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
	
:0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
	
;0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
	
<0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
 
0
1
)2
*3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
0
 1
+2
,3*

0
 1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
!0
"1
-2
.3*

!0
"1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
#0
$1
/2
03*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
%0
&1
12
23*

%0
&1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
'0
(1
32
43*

'0
(1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
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
	
=0* 
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
d^
VARIABLE_VALUESGD/m/cdnn/dense_3/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUESGD/m/cdnn/dense_3/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUESGD/m/cdnn/dense_4/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUESGD/m/cdnn/dense_4/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUESGD/m/cdnn/dense_5/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUESGD/m/cdnn/dense_5/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$SGD/m/cdnn/batch_normalization/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#SGD/m/cdnn/batch_normalization/beta2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_1/gamma2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_2/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_2/beta2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_3/gamma2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_3/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_4/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_4/beta2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_5/gamma2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_5/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUESGD/m/cdnn/dense_6/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUESGD/m/cdnn/dense_6/bias2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
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
�0
�1*
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
	
70* 
* 
* 
* 
* 
* 
* 
	
80* 
* 
* 
* 
* 
* 
* 
	
90* 
* 
* 
* 
* 
* 
* 
	
:0* 
* 
* 
* 
* 
* 
* 
	
;0* 
* 
* 
* 
* 
* 
* 
	
<0* 
* 
* 
* 

)0
*1*
* 
* 
* 
* 
* 
* 
* 
* 

+0
,1*
* 
* 
* 
* 
* 
* 
* 
* 

-0
.1*
* 
* 
* 
* 
* 
* 
* 
* 

/0
01*
* 
* 
* 
* 
* 
* 
* 
* 

10
21*
* 
* 
* 
* 
* 
* 
* 
* 

30
41*
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%cdnn/dense/kernel/Read/ReadVariableOp#cdnn/dense/bias/Read/ReadVariableOp'cdnn/dense_1/kernel/Read/ReadVariableOp%cdnn/dense_1/bias/Read/ReadVariableOp'cdnn/dense_2/kernel/Read/ReadVariableOp%cdnn/dense_2/bias/Read/ReadVariableOp'cdnn/dense_3/kernel/Read/ReadVariableOp%cdnn/dense_3/bias/Read/ReadVariableOp'cdnn/dense_4/kernel/Read/ReadVariableOp%cdnn/dense_4/bias/Read/ReadVariableOp'cdnn/dense_5/kernel/Read/ReadVariableOp%cdnn/dense_5/bias/Read/ReadVariableOp2cdnn/batch_normalization/gamma/Read/ReadVariableOp1cdnn/batch_normalization/beta/Read/ReadVariableOp4cdnn/batch_normalization_1/gamma/Read/ReadVariableOp3cdnn/batch_normalization_1/beta/Read/ReadVariableOp4cdnn/batch_normalization_2/gamma/Read/ReadVariableOp3cdnn/batch_normalization_2/beta/Read/ReadVariableOp4cdnn/batch_normalization_3/gamma/Read/ReadVariableOp3cdnn/batch_normalization_3/beta/Read/ReadVariableOp4cdnn/batch_normalization_4/gamma/Read/ReadVariableOp3cdnn/batch_normalization_4/beta/Read/ReadVariableOp4cdnn/batch_normalization_5/gamma/Read/ReadVariableOp3cdnn/batch_normalization_5/beta/Read/ReadVariableOp8cdnn/batch_normalization/moving_mean/Read/ReadVariableOp<cdnn/batch_normalization/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_1/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_1/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_2/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_2/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_3/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_3/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_4/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_4/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_5/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_5/moving_variance/Read/ReadVariableOp'cdnn/dense_6/kernel/Read/ReadVariableOp%cdnn/dense_6/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp+SGD/m/cdnn/dense/kernel/Read/ReadVariableOp)SGD/m/cdnn/dense/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_1/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_1/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_2/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_2/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_3/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_3/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_4/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_4/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_5/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_5/bias/Read/ReadVariableOp8SGD/m/cdnn/batch_normalization/gamma/Read/ReadVariableOp7SGD/m/cdnn/batch_normalization/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_1/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_1/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_2/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_2/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_3/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_3/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_4/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_4/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_5/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_5/beta/Read/ReadVariableOp-SGD/m/cdnn/dense_6/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_6/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*S
TinL
J2H	*
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
__inference__traced_save_183554
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecdnn/dense/kernelcdnn/dense/biascdnn/dense_1/kernelcdnn/dense_1/biascdnn/dense_2/kernelcdnn/dense_2/biascdnn/dense_3/kernelcdnn/dense_3/biascdnn/dense_4/kernelcdnn/dense_4/biascdnn/dense_5/kernelcdnn/dense_5/biascdnn/batch_normalization/gammacdnn/batch_normalization/beta cdnn/batch_normalization_1/gammacdnn/batch_normalization_1/beta cdnn/batch_normalization_2/gammacdnn/batch_normalization_2/beta cdnn/batch_normalization_3/gammacdnn/batch_normalization_3/beta cdnn/batch_normalization_4/gammacdnn/batch_normalization_4/beta cdnn/batch_normalization_5/gammacdnn/batch_normalization_5/beta$cdnn/batch_normalization/moving_mean(cdnn/batch_normalization/moving_variance&cdnn/batch_normalization_1/moving_mean*cdnn/batch_normalization_1/moving_variance&cdnn/batch_normalization_2/moving_mean*cdnn/batch_normalization_2/moving_variance&cdnn/batch_normalization_3/moving_mean*cdnn/batch_normalization_3/moving_variance&cdnn/batch_normalization_4/moving_mean*cdnn/batch_normalization_4/moving_variance&cdnn/batch_normalization_5/moving_mean*cdnn/batch_normalization_5/moving_variancecdnn/dense_6/kernelcdnn/dense_6/bias	iterationlearning_rateSGD/m/cdnn/dense/kernelSGD/m/cdnn/dense/biasSGD/m/cdnn/dense_1/kernelSGD/m/cdnn/dense_1/biasSGD/m/cdnn/dense_2/kernelSGD/m/cdnn/dense_2/biasSGD/m/cdnn/dense_3/kernelSGD/m/cdnn/dense_3/biasSGD/m/cdnn/dense_4/kernelSGD/m/cdnn/dense_4/biasSGD/m/cdnn/dense_5/kernelSGD/m/cdnn/dense_5/bias$SGD/m/cdnn/batch_normalization/gamma#SGD/m/cdnn/batch_normalization/beta&SGD/m/cdnn/batch_normalization_1/gamma%SGD/m/cdnn/batch_normalization_1/beta&SGD/m/cdnn/batch_normalization_2/gamma%SGD/m/cdnn/batch_normalization_2/beta&SGD/m/cdnn/batch_normalization_3/gamma%SGD/m/cdnn/batch_normalization_3/beta&SGD/m/cdnn/batch_normalization_4/gamma%SGD/m/cdnn/batch_normalization_4/beta&SGD/m/cdnn/batch_normalization_5/gamma%SGD/m/cdnn/batch_normalization_5/betaSGD/m/cdnn/dense_6/kernelSGD/m/cdnn/dense_6/biastotal_1count_1totalcount*R
TinK
I2G*
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
"__inference__traced_restore_183774��
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183287

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
�
__inference_loss_fn_4_182595P
>cdnn_dense_4_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_4_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_4/kernel/Regularizer/L2LossL2Loss=cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_4/kernel/Regularizer/mulMul.cdnn/dense_4/kernel/Regularizer/mul/x:output:0/cdnn/dense_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'cdnn/dense_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
6__inference_batch_normalization_2_layer_call_fn_183014

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_180480o
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
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_180726

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
�
�!
__inference__traced_save_183554
file_prefix0
,savev2_cdnn_dense_kernel_read_readvariableop.
*savev2_cdnn_dense_bias_read_readvariableop2
.savev2_cdnn_dense_1_kernel_read_readvariableop0
,savev2_cdnn_dense_1_bias_read_readvariableop2
.savev2_cdnn_dense_2_kernel_read_readvariableop0
,savev2_cdnn_dense_2_bias_read_readvariableop2
.savev2_cdnn_dense_3_kernel_read_readvariableop0
,savev2_cdnn_dense_3_bias_read_readvariableop2
.savev2_cdnn_dense_4_kernel_read_readvariableop0
,savev2_cdnn_dense_4_bias_read_readvariableop2
.savev2_cdnn_dense_5_kernel_read_readvariableop0
,savev2_cdnn_dense_5_bias_read_readvariableop=
9savev2_cdnn_batch_normalization_gamma_read_readvariableop<
8savev2_cdnn_batch_normalization_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_1_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_1_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_2_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_2_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_3_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_3_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_4_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_4_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_5_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_5_beta_read_readvariableopC
?savev2_cdnn_batch_normalization_moving_mean_read_readvariableopG
Csavev2_cdnn_batch_normalization_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_1_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_1_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_2_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_2_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_3_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_3_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_4_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_4_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_5_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_5_moving_variance_read_readvariableop2
.savev2_cdnn_dense_6_kernel_read_readvariableop0
,savev2_cdnn_dense_6_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop6
2savev2_sgd_m_cdnn_dense_kernel_read_readvariableop4
0savev2_sgd_m_cdnn_dense_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_1_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_1_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_2_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_2_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_3_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_3_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_4_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_4_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_5_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_5_bias_read_readvariableopC
?savev2_sgd_m_cdnn_batch_normalization_gamma_read_readvariableopB
>savev2_sgd_m_cdnn_batch_normalization_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_1_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_1_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_2_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_2_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_3_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_3_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_4_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_4_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_5_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_5_beta_read_readvariableop8
4savev2_sgd_m_cdnn_dense_6_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_6_bias_read_readvariableop&
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_cdnn_dense_kernel_read_readvariableop*savev2_cdnn_dense_bias_read_readvariableop.savev2_cdnn_dense_1_kernel_read_readvariableop,savev2_cdnn_dense_1_bias_read_readvariableop.savev2_cdnn_dense_2_kernel_read_readvariableop,savev2_cdnn_dense_2_bias_read_readvariableop.savev2_cdnn_dense_3_kernel_read_readvariableop,savev2_cdnn_dense_3_bias_read_readvariableop.savev2_cdnn_dense_4_kernel_read_readvariableop,savev2_cdnn_dense_4_bias_read_readvariableop.savev2_cdnn_dense_5_kernel_read_readvariableop,savev2_cdnn_dense_5_bias_read_readvariableop9savev2_cdnn_batch_normalization_gamma_read_readvariableop8savev2_cdnn_batch_normalization_beta_read_readvariableop;savev2_cdnn_batch_normalization_1_gamma_read_readvariableop:savev2_cdnn_batch_normalization_1_beta_read_readvariableop;savev2_cdnn_batch_normalization_2_gamma_read_readvariableop:savev2_cdnn_batch_normalization_2_beta_read_readvariableop;savev2_cdnn_batch_normalization_3_gamma_read_readvariableop:savev2_cdnn_batch_normalization_3_beta_read_readvariableop;savev2_cdnn_batch_normalization_4_gamma_read_readvariableop:savev2_cdnn_batch_normalization_4_beta_read_readvariableop;savev2_cdnn_batch_normalization_5_gamma_read_readvariableop:savev2_cdnn_batch_normalization_5_beta_read_readvariableop?savev2_cdnn_batch_normalization_moving_mean_read_readvariableopCsavev2_cdnn_batch_normalization_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_1_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_1_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_2_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_2_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_3_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_3_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_4_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_4_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_5_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_5_moving_variance_read_readvariableop.savev2_cdnn_dense_6_kernel_read_readvariableop,savev2_cdnn_dense_6_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop2savev2_sgd_m_cdnn_dense_kernel_read_readvariableop0savev2_sgd_m_cdnn_dense_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_1_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_1_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_2_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_2_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_3_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_3_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_4_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_4_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_5_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_5_bias_read_readvariableop?savev2_sgd_m_cdnn_batch_normalization_gamma_read_readvariableop>savev2_sgd_m_cdnn_batch_normalization_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_1_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_1_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_2_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_2_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_3_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_3_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_4_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_4_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_5_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_5_beta_read_readvariableop4savev2_sgd_m_cdnn_dense_6_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_6_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *U
dtypesK
I2G	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::: : ::::::::::::::::::::::::::: : : : : 2(
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
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 
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
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 
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
:: #

_output_shapes
:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :$) 

_output_shapes

:: *

_output_shapes
::$+ 

_output_shapes

:: ,

_output_shapes
::$- 

_output_shapes

:: .

_output_shapes
::$/ 

_output_shapes

:: 0

_output_shapes
::$1 

_output_shapes

:: 2

_output_shapes
::$3 

_output_shapes

:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
::$A 

_output_shapes

:: B

_output_shapes
::C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: 
�
�
4__inference_batch_normalization_layer_call_fn_182854

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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_180316o
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
��
�/
"__inference__traced_restore_183774
file_prefix4
"assignvariableop_cdnn_dense_kernel:0
"assignvariableop_1_cdnn_dense_bias:8
&assignvariableop_2_cdnn_dense_1_kernel:2
$assignvariableop_3_cdnn_dense_1_bias:8
&assignvariableop_4_cdnn_dense_2_kernel:2
$assignvariableop_5_cdnn_dense_2_bias:8
&assignvariableop_6_cdnn_dense_3_kernel:2
$assignvariableop_7_cdnn_dense_3_bias:8
&assignvariableop_8_cdnn_dense_4_kernel:2
$assignvariableop_9_cdnn_dense_4_bias:9
'assignvariableop_10_cdnn_dense_5_kernel:3
%assignvariableop_11_cdnn_dense_5_bias:@
2assignvariableop_12_cdnn_batch_normalization_gamma:?
1assignvariableop_13_cdnn_batch_normalization_beta:B
4assignvariableop_14_cdnn_batch_normalization_1_gamma:A
3assignvariableop_15_cdnn_batch_normalization_1_beta:B
4assignvariableop_16_cdnn_batch_normalization_2_gamma:A
3assignvariableop_17_cdnn_batch_normalization_2_beta:B
4assignvariableop_18_cdnn_batch_normalization_3_gamma:A
3assignvariableop_19_cdnn_batch_normalization_3_beta:B
4assignvariableop_20_cdnn_batch_normalization_4_gamma:A
3assignvariableop_21_cdnn_batch_normalization_4_beta:B
4assignvariableop_22_cdnn_batch_normalization_5_gamma:A
3assignvariableop_23_cdnn_batch_normalization_5_beta:F
8assignvariableop_24_cdnn_batch_normalization_moving_mean:J
<assignvariableop_25_cdnn_batch_normalization_moving_variance:H
:assignvariableop_26_cdnn_batch_normalization_1_moving_mean:L
>assignvariableop_27_cdnn_batch_normalization_1_moving_variance:H
:assignvariableop_28_cdnn_batch_normalization_2_moving_mean:L
>assignvariableop_29_cdnn_batch_normalization_2_moving_variance:H
:assignvariableop_30_cdnn_batch_normalization_3_moving_mean:L
>assignvariableop_31_cdnn_batch_normalization_3_moving_variance:H
:assignvariableop_32_cdnn_batch_normalization_4_moving_mean:L
>assignvariableop_33_cdnn_batch_normalization_4_moving_variance:H
:assignvariableop_34_cdnn_batch_normalization_5_moving_mean:L
>assignvariableop_35_cdnn_batch_normalization_5_moving_variance:9
'assignvariableop_36_cdnn_dense_6_kernel:3
%assignvariableop_37_cdnn_dense_6_bias:'
assignvariableop_38_iteration:	 +
!assignvariableop_39_learning_rate: =
+assignvariableop_40_sgd_m_cdnn_dense_kernel:7
)assignvariableop_41_sgd_m_cdnn_dense_bias:?
-assignvariableop_42_sgd_m_cdnn_dense_1_kernel:9
+assignvariableop_43_sgd_m_cdnn_dense_1_bias:?
-assignvariableop_44_sgd_m_cdnn_dense_2_kernel:9
+assignvariableop_45_sgd_m_cdnn_dense_2_bias:?
-assignvariableop_46_sgd_m_cdnn_dense_3_kernel:9
+assignvariableop_47_sgd_m_cdnn_dense_3_bias:?
-assignvariableop_48_sgd_m_cdnn_dense_4_kernel:9
+assignvariableop_49_sgd_m_cdnn_dense_4_bias:?
-assignvariableop_50_sgd_m_cdnn_dense_5_kernel:9
+assignvariableop_51_sgd_m_cdnn_dense_5_bias:F
8assignvariableop_52_sgd_m_cdnn_batch_normalization_gamma:E
7assignvariableop_53_sgd_m_cdnn_batch_normalization_beta:H
:assignvariableop_54_sgd_m_cdnn_batch_normalization_1_gamma:G
9assignvariableop_55_sgd_m_cdnn_batch_normalization_1_beta:H
:assignvariableop_56_sgd_m_cdnn_batch_normalization_2_gamma:G
9assignvariableop_57_sgd_m_cdnn_batch_normalization_2_beta:H
:assignvariableop_58_sgd_m_cdnn_batch_normalization_3_gamma:G
9assignvariableop_59_sgd_m_cdnn_batch_normalization_3_beta:H
:assignvariableop_60_sgd_m_cdnn_batch_normalization_4_gamma:G
9assignvariableop_61_sgd_m_cdnn_batch_normalization_4_beta:H
:assignvariableop_62_sgd_m_cdnn_batch_normalization_5_gamma:G
9assignvariableop_63_sgd_m_cdnn_batch_normalization_5_beta:?
-assignvariableop_64_sgd_m_cdnn_dense_6_kernel:9
+assignvariableop_65_sgd_m_cdnn_dense_6_bias:%
assignvariableop_66_total_1: %
assignvariableop_67_count_1: #
assignvariableop_68_total: #
assignvariableop_69_count: 
identity_71��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*U
dtypesK
I2G	[
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
AssignVariableOp_6AssignVariableOp&assignvariableop_6_cdnn_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_cdnn_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_cdnn_dense_4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp$assignvariableop_9_cdnn_dense_4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_cdnn_dense_5_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_cdnn_dense_5_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp2assignvariableop_12_cdnn_batch_normalization_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp1assignvariableop_13_cdnn_batch_normalization_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp4assignvariableop_14_cdnn_batch_normalization_1_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp3assignvariableop_15_cdnn_batch_normalization_1_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp4assignvariableop_16_cdnn_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp3assignvariableop_17_cdnn_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_cdnn_batch_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp3assignvariableop_19_cdnn_batch_normalization_3_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp4assignvariableop_20_cdnn_batch_normalization_4_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp3assignvariableop_21_cdnn_batch_normalization_4_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_cdnn_batch_normalization_5_gammaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp3assignvariableop_23_cdnn_batch_normalization_5_betaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp8assignvariableop_24_cdnn_batch_normalization_moving_meanIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp<assignvariableop_25_cdnn_batch_normalization_moving_varianceIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp:assignvariableop_26_cdnn_batch_normalization_1_moving_meanIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp>assignvariableop_27_cdnn_batch_normalization_1_moving_varianceIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp:assignvariableop_28_cdnn_batch_normalization_2_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp>assignvariableop_29_cdnn_batch_normalization_2_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp:assignvariableop_30_cdnn_batch_normalization_3_moving_meanIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp>assignvariableop_31_cdnn_batch_normalization_3_moving_varianceIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp:assignvariableop_32_cdnn_batch_normalization_4_moving_meanIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp>assignvariableop_33_cdnn_batch_normalization_4_moving_varianceIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp:assignvariableop_34_cdnn_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp>assignvariableop_35_cdnn_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_cdnn_dense_6_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp%assignvariableop_37_cdnn_dense_6_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_iterationIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp!assignvariableop_39_learning_rateIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp+assignvariableop_40_sgd_m_cdnn_dense_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp)assignvariableop_41_sgd_m_cdnn_dense_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp-assignvariableop_42_sgd_m_cdnn_dense_1_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_sgd_m_cdnn_dense_1_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp-assignvariableop_44_sgd_m_cdnn_dense_2_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_sgd_m_cdnn_dense_2_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp-assignvariableop_46_sgd_m_cdnn_dense_3_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_sgd_m_cdnn_dense_3_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp-assignvariableop_48_sgd_m_cdnn_dense_4_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp+assignvariableop_49_sgd_m_cdnn_dense_4_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp-assignvariableop_50_sgd_m_cdnn_dense_5_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp+assignvariableop_51_sgd_m_cdnn_dense_5_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp8assignvariableop_52_sgd_m_cdnn_batch_normalization_gammaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp7assignvariableop_53_sgd_m_cdnn_batch_normalization_betaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp:assignvariableop_54_sgd_m_cdnn_batch_normalization_1_gammaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp9assignvariableop_55_sgd_m_cdnn_batch_normalization_1_betaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp:assignvariableop_56_sgd_m_cdnn_batch_normalization_2_gammaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp9assignvariableop_57_sgd_m_cdnn_batch_normalization_2_betaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp:assignvariableop_58_sgd_m_cdnn_batch_normalization_3_gammaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp9assignvariableop_59_sgd_m_cdnn_batch_normalization_3_betaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp:assignvariableop_60_sgd_m_cdnn_batch_normalization_4_gammaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp9assignvariableop_61_sgd_m_cdnn_batch_normalization_4_betaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp:assignvariableop_62_sgd_m_cdnn_batch_normalization_5_gammaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp9assignvariableop_63_sgd_m_cdnn_batch_normalization_5_betaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp-assignvariableop_64_sgd_m_cdnn_dense_6_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp+assignvariableop_65_sgd_m_cdnn_dense_6_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOpassignvariableop_66_total_1Identity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpassignvariableop_67_count_1Identity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpassignvariableop_68_totalIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOpassignvariableop_69_countIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_70Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_71IdentityIdentity_70:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_71Identity_71:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
__inference_loss_fn_2_182577P
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
 *�ŧ7�
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
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182967

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
�
G
+__inference_activation_layer_call_fn_182692

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
F__inference_activation_layer_call_and_return_conditional_losses_181003`
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
�
�
&__inference_dense_layer_call_fn_182706

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
A__inference_dense_layer_call_and_return_conditional_losses_180806o
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
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_180926

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_4/kernel/Regularizer/L2LossL2Loss=cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_4/kernel/Regularizer/mulMul.cdnn/dense_4/kernel/Regularizer/mul/x:output:0/cdnn/dense_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_182947

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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_180445o
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
�$
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183161

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
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_182793

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
 *�ŧ7�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
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
�	
�
__inference_loss_fn_6_182613P
>cdnn_dense_6_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_6_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_6/kernel/Regularizer/L2LossL2Loss=cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_6/kernel/Regularizer/mulMul.cdnn/dense_6/kernel/Regularizer/mul/x:output:0/cdnn/dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'cdnn/dense_6/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp
�$
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_180609

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
�
�
A__inference_dense_layer_call_and_return_conditional_losses_182721

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
 *�ŧ7�
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
�$
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_180773

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
�
�
@__inference_cdnn_layer_call_and_return_conditional_losses_181701
input_1
dense_181581:
dense_181583:(
batch_normalization_181586:(
batch_normalization_181588:(
batch_normalization_181590:(
batch_normalization_181592: 
dense_1_181595:
dense_1_181597:*
batch_normalization_1_181600:*
batch_normalization_1_181602:*
batch_normalization_1_181604:*
batch_normalization_1_181606: 
dense_2_181609:
dense_2_181611:*
batch_normalization_2_181614:*
batch_normalization_2_181616:*
batch_normalization_2_181618:*
batch_normalization_2_181620: 
dense_3_181623:
dense_3_181625:*
batch_normalization_3_181628:*
batch_normalization_3_181630:*
batch_normalization_3_181632:*
batch_normalization_3_181634: 
dense_4_181637:
dense_4_181639:*
batch_normalization_4_181642:*
batch_normalization_4_181644:*
batch_normalization_4_181646:*
batch_normalization_4_181648: 
dense_5_181651:
dense_5_181653:*
batch_normalization_5_181656:*
batch_normalization_5_181658:*
batch_normalization_5_181660:*
batch_normalization_5_181662: 
dense_6_181666:
dense_6_181668:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_181581dense_181583*
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
A__inference_dense_layer_call_and_return_conditional_losses_180806�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_181586batch_normalization_181588batch_normalization_181590batch_normalization_181592*
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_180316�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_181595dense_1_181597*
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
C__inference_dense_1_layer_call_and_return_conditional_losses_180836�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_181600batch_normalization_1_181602batch_normalization_1_181604batch_normalization_1_181606*
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_180398�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_181609dense_2_181611*
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
C__inference_dense_2_layer_call_and_return_conditional_losses_180866�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_181614batch_normalization_2_181616batch_normalization_2_181618batch_normalization_2_181620*
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_180480�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_181623dense_3_181625*
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
C__inference_dense_3_layer_call_and_return_conditional_losses_180896�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_3_181628batch_normalization_3_181630batch_normalization_3_181632batch_normalization_3_181634*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_180562�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_181637dense_4_181639*
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
C__inference_dense_4_layer_call_and_return_conditional_losses_180926�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_4_181642batch_normalization_4_181644batch_normalization_4_181646batch_normalization_4_181648*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_180644�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_181651dense_5_181653*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_180956�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_5_181656batch_normalization_5_181658batch_normalization_5_181660batch_normalization_5_181662*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_180726�
dropout/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_180976�
dense_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_6_181666dense_6_181668*
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
C__inference_dense_6_layer_call_and_return_conditional_losses_180992�
activation/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
F__inference_activation_layer_call_and_return_conditional_losses_181003�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_181581*
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
 *�ŧ7�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_181595*
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
 *�ŧ7�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_181609*
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
 *�ŧ7�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_181623*
_output_shapes

:*
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
 *�ŧ7�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_4_181637*
_output_shapes

:*
dtype0�
&cdnn/dense_4/kernel/Regularizer/L2LossL2Loss=cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_4/kernel/Regularizer/mulMul.cdnn/dense_4/kernel/Regularizer/mul/x:output:0/cdnn/dense_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_5_181651*
_output_shapes

:*
dtype0�
&cdnn/dense_5/kernel/Regularizer/L2LossL2Loss=cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_5/kernel/Regularizer/mulMul.cdnn/dense_5/kernel/Regularizer/mul/x:output:0/cdnn/dense_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_181666*
_output_shapes

:*
dtype0�
&cdnn/dense_6/kernel/Regularizer/L2LossL2Loss=cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_6/kernel/Regularizer/mulMul.cdnn/dense_6/kernel/Regularizer/mul/x:output:0/cdnn/dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�$
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_180527

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
�
�
6__inference_batch_normalization_4_layer_call_fn_183174

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_180644o
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
�
D
(__inference_dropout_layer_call_fn_182642

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
C__inference_dropout_layer_call_and_return_conditional_losses_180976`
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
�
b
F__inference_activation_layer_call_and_return_conditional_losses_181003

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
�
�
(__inference_dense_6_layer_call_fn_182673

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
C__inference_dense_6_layer_call_and_return_conditional_losses_180992o
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
��
�"
!__inference__wrapped_model_180292
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
+cdnn_dense_3_matmul_readvariableop_resource::
,cdnn_dense_3_biasadd_readvariableop_resource:E
7cdnn_batch_normalization_3_cast_readvariableop_resource:G
9cdnn_batch_normalization_3_cast_1_readvariableop_resource:G
9cdnn_batch_normalization_3_cast_2_readvariableop_resource:G
9cdnn_batch_normalization_3_cast_3_readvariableop_resource:=
+cdnn_dense_4_matmul_readvariableop_resource::
,cdnn_dense_4_biasadd_readvariableop_resource:E
7cdnn_batch_normalization_4_cast_readvariableop_resource:G
9cdnn_batch_normalization_4_cast_1_readvariableop_resource:G
9cdnn_batch_normalization_4_cast_2_readvariableop_resource:G
9cdnn_batch_normalization_4_cast_3_readvariableop_resource:=
+cdnn_dense_5_matmul_readvariableop_resource::
,cdnn_dense_5_biasadd_readvariableop_resource:E
7cdnn_batch_normalization_5_cast_readvariableop_resource:G
9cdnn_batch_normalization_5_cast_1_readvariableop_resource:G
9cdnn_batch_normalization_5_cast_2_readvariableop_resource:G
9cdnn_batch_normalization_5_cast_3_readvariableop_resource:=
+cdnn_dense_6_matmul_readvariableop_resource::
,cdnn_dense_6_biasadd_readvariableop_resource:
identity��,cdnn/batch_normalization/Cast/ReadVariableOp�.cdnn/batch_normalization/Cast_1/ReadVariableOp�.cdnn/batch_normalization/Cast_2/ReadVariableOp�.cdnn/batch_normalization/Cast_3/ReadVariableOp�.cdnn/batch_normalization_1/Cast/ReadVariableOp�0cdnn/batch_normalization_1/Cast_1/ReadVariableOp�0cdnn/batch_normalization_1/Cast_2/ReadVariableOp�0cdnn/batch_normalization_1/Cast_3/ReadVariableOp�.cdnn/batch_normalization_2/Cast/ReadVariableOp�0cdnn/batch_normalization_2/Cast_1/ReadVariableOp�0cdnn/batch_normalization_2/Cast_2/ReadVariableOp�0cdnn/batch_normalization_2/Cast_3/ReadVariableOp�.cdnn/batch_normalization_3/Cast/ReadVariableOp�0cdnn/batch_normalization_3/Cast_1/ReadVariableOp�0cdnn/batch_normalization_3/Cast_2/ReadVariableOp�0cdnn/batch_normalization_3/Cast_3/ReadVariableOp�.cdnn/batch_normalization_4/Cast/ReadVariableOp�0cdnn/batch_normalization_4/Cast_1/ReadVariableOp�0cdnn/batch_normalization_4/Cast_2/ReadVariableOp�0cdnn/batch_normalization_4/Cast_3/ReadVariableOp�.cdnn/batch_normalization_5/Cast/ReadVariableOp�0cdnn/batch_normalization_5/Cast_1/ReadVariableOp�0cdnn/batch_normalization_5/Cast_2/ReadVariableOp�0cdnn/batch_normalization_5/Cast_3/ReadVariableOp�!cdnn/dense/BiasAdd/ReadVariableOp� cdnn/dense/MatMul/ReadVariableOp�#cdnn/dense_1/BiasAdd/ReadVariableOp�"cdnn/dense_1/MatMul/ReadVariableOp�#cdnn/dense_2/BiasAdd/ReadVariableOp�"cdnn/dense_2/MatMul/ReadVariableOp�#cdnn/dense_3/BiasAdd/ReadVariableOp�"cdnn/dense_3/MatMul/ReadVariableOp�#cdnn/dense_4/BiasAdd/ReadVariableOp�"cdnn/dense_4/MatMul/ReadVariableOp�#cdnn/dense_5/BiasAdd/ReadVariableOp�"cdnn/dense_5/MatMul/ReadVariableOp�#cdnn/dense_6/BiasAdd/ReadVariableOp�"cdnn/dense_6/MatMul/ReadVariableOp�
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
"cdnn/dense_3/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_3/MatMulMatMul.cdnn/batch_normalization_2/batchnorm/add_1:z:0*cdnn/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_3/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_3/BiasAddBiasAddcdnn/dense_3/MatMul:product:0+cdnn/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
cdnn/dense_3/ReluRelucdnn/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.cdnn/batch_normalization_3/Cast/ReadVariableOpReadVariableOp7cdnn_batch_normalization_3_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp9cdnn_batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_3/Cast_2/ReadVariableOpReadVariableOp9cdnn_batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_3/Cast_3/ReadVariableOpReadVariableOp9cdnn_batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0o
*cdnn/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(cdnn/batch_normalization_3/batchnorm/addAddV28cdnn/batch_normalization_3/Cast_1/ReadVariableOp:value:03cdnn/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_3/batchnorm/RsqrtRsqrt,cdnn/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_3/batchnorm/mulMul.cdnn/batch_normalization_3/batchnorm/Rsqrt:y:08cdnn/batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_3/batchnorm/mul_1Mulcdnn/dense_3/Relu:activations:0,cdnn/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
*cdnn/batch_normalization_3/batchnorm/mul_2Mul6cdnn/batch_normalization_3/Cast/ReadVariableOp:value:0,cdnn/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_3/batchnorm/subSub8cdnn/batch_normalization_3/Cast_2/ReadVariableOp:value:0.cdnn/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_3/batchnorm/add_1AddV2.cdnn/batch_normalization_3/batchnorm/mul_1:z:0,cdnn/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
"cdnn/dense_4/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_4/MatMulMatMul.cdnn/batch_normalization_3/batchnorm/add_1:z:0*cdnn/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_4/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_4/BiasAddBiasAddcdnn/dense_4/MatMul:product:0+cdnn/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
cdnn/dense_4/ReluRelucdnn/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.cdnn/batch_normalization_4/Cast/ReadVariableOpReadVariableOp7cdnn_batch_normalization_4_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp9cdnn_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp9cdnn_batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp9cdnn_batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0o
*cdnn/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(cdnn/batch_normalization_4/batchnorm/addAddV28cdnn/batch_normalization_4/Cast_1/ReadVariableOp:value:03cdnn/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_4/batchnorm/RsqrtRsqrt,cdnn/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_4/batchnorm/mulMul.cdnn/batch_normalization_4/batchnorm/Rsqrt:y:08cdnn/batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_4/batchnorm/mul_1Mulcdnn/dense_4/Relu:activations:0,cdnn/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
*cdnn/batch_normalization_4/batchnorm/mul_2Mul6cdnn/batch_normalization_4/Cast/ReadVariableOp:value:0,cdnn/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_4/batchnorm/subSub8cdnn/batch_normalization_4/Cast_2/ReadVariableOp:value:0.cdnn/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_4/batchnorm/add_1AddV2.cdnn/batch_normalization_4/batchnorm/mul_1:z:0,cdnn/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
"cdnn/dense_5/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_5/MatMulMatMul.cdnn/batch_normalization_4/batchnorm/add_1:z:0*cdnn/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_5/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_5/BiasAddBiasAddcdnn/dense_5/MatMul:product:0+cdnn/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
cdnn/dense_5/ReluRelucdnn/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.cdnn/batch_normalization_5/Cast/ReadVariableOpReadVariableOp7cdnn_batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp9cdnn_batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp9cdnn_batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp9cdnn_batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0o
*cdnn/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(cdnn/batch_normalization_5/batchnorm/addAddV28cdnn/batch_normalization_5/Cast_1/ReadVariableOp:value:03cdnn/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_5/batchnorm/RsqrtRsqrt,cdnn/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_5/batchnorm/mulMul.cdnn/batch_normalization_5/batchnorm/Rsqrt:y:08cdnn/batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_5/batchnorm/mul_1Mulcdnn/dense_5/Relu:activations:0,cdnn/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
*cdnn/batch_normalization_5/batchnorm/mul_2Mul6cdnn/batch_normalization_5/Cast/ReadVariableOp:value:0,cdnn/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_5/batchnorm/subSub8cdnn/batch_normalization_5/Cast_2/ReadVariableOp:value:0.cdnn/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_5/batchnorm/add_1AddV2.cdnn/batch_normalization_5/batchnorm/mul_1:z:0,cdnn/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
cdnn/dropout/IdentityIdentity.cdnn/batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:����������
"cdnn/dense_6/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_6/MatMulMatMulcdnn/dropout/Identity:output:0*cdnn/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_6/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_6/BiasAddBiasAddcdnn/dense_6/MatMul:product:0+cdnn/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s
cdnn/activation/SigmoidSigmoidcdnn/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentitycdnn/activation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^cdnn/batch_normalization/Cast/ReadVariableOp/^cdnn/batch_normalization/Cast_1/ReadVariableOp/^cdnn/batch_normalization/Cast_2/ReadVariableOp/^cdnn/batch_normalization/Cast_3/ReadVariableOp/^cdnn/batch_normalization_1/Cast/ReadVariableOp1^cdnn/batch_normalization_1/Cast_1/ReadVariableOp1^cdnn/batch_normalization_1/Cast_2/ReadVariableOp1^cdnn/batch_normalization_1/Cast_3/ReadVariableOp/^cdnn/batch_normalization_2/Cast/ReadVariableOp1^cdnn/batch_normalization_2/Cast_1/ReadVariableOp1^cdnn/batch_normalization_2/Cast_2/ReadVariableOp1^cdnn/batch_normalization_2/Cast_3/ReadVariableOp/^cdnn/batch_normalization_3/Cast/ReadVariableOp1^cdnn/batch_normalization_3/Cast_1/ReadVariableOp1^cdnn/batch_normalization_3/Cast_2/ReadVariableOp1^cdnn/batch_normalization_3/Cast_3/ReadVariableOp/^cdnn/batch_normalization_4/Cast/ReadVariableOp1^cdnn/batch_normalization_4/Cast_1/ReadVariableOp1^cdnn/batch_normalization_4/Cast_2/ReadVariableOp1^cdnn/batch_normalization_4/Cast_3/ReadVariableOp/^cdnn/batch_normalization_5/Cast/ReadVariableOp1^cdnn/batch_normalization_5/Cast_1/ReadVariableOp1^cdnn/batch_normalization_5/Cast_2/ReadVariableOp1^cdnn/batch_normalization_5/Cast_3/ReadVariableOp"^cdnn/dense/BiasAdd/ReadVariableOp!^cdnn/dense/MatMul/ReadVariableOp$^cdnn/dense_1/BiasAdd/ReadVariableOp#^cdnn/dense_1/MatMul/ReadVariableOp$^cdnn/dense_2/BiasAdd/ReadVariableOp#^cdnn/dense_2/MatMul/ReadVariableOp$^cdnn/dense_3/BiasAdd/ReadVariableOp#^cdnn/dense_3/MatMul/ReadVariableOp$^cdnn/dense_4/BiasAdd/ReadVariableOp#^cdnn/dense_4/MatMul/ReadVariableOp$^cdnn/dense_5/BiasAdd/ReadVariableOp#^cdnn/dense_5/MatMul/ReadVariableOp$^cdnn/dense_6/BiasAdd/ReadVariableOp#^cdnn/dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
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
0cdnn/batch_normalization_2/Cast_3/ReadVariableOp0cdnn/batch_normalization_2/Cast_3/ReadVariableOp2`
.cdnn/batch_normalization_3/Cast/ReadVariableOp.cdnn/batch_normalization_3/Cast/ReadVariableOp2d
0cdnn/batch_normalization_3/Cast_1/ReadVariableOp0cdnn/batch_normalization_3/Cast_1/ReadVariableOp2d
0cdnn/batch_normalization_3/Cast_2/ReadVariableOp0cdnn/batch_normalization_3/Cast_2/ReadVariableOp2d
0cdnn/batch_normalization_3/Cast_3/ReadVariableOp0cdnn/batch_normalization_3/Cast_3/ReadVariableOp2`
.cdnn/batch_normalization_4/Cast/ReadVariableOp.cdnn/batch_normalization_4/Cast/ReadVariableOp2d
0cdnn/batch_normalization_4/Cast_1/ReadVariableOp0cdnn/batch_normalization_4/Cast_1/ReadVariableOp2d
0cdnn/batch_normalization_4/Cast_2/ReadVariableOp0cdnn/batch_normalization_4/Cast_2/ReadVariableOp2d
0cdnn/batch_normalization_4/Cast_3/ReadVariableOp0cdnn/batch_normalization_4/Cast_3/ReadVariableOp2`
.cdnn/batch_normalization_5/Cast/ReadVariableOp.cdnn/batch_normalization_5/Cast/ReadVariableOp2d
0cdnn/batch_normalization_5/Cast_1/ReadVariableOp0cdnn/batch_normalization_5/Cast_1/ReadVariableOp2d
0cdnn/batch_normalization_5/Cast_2/ReadVariableOp0cdnn/batch_normalization_5/Cast_2/ReadVariableOp2d
0cdnn/batch_normalization_5/Cast_3/ReadVariableOp0cdnn/batch_normalization_5/Cast_3/ReadVariableOp2F
!cdnn/dense/BiasAdd/ReadVariableOp!cdnn/dense/BiasAdd/ReadVariableOp2D
 cdnn/dense/MatMul/ReadVariableOp cdnn/dense/MatMul/ReadVariableOp2J
#cdnn/dense_1/BiasAdd/ReadVariableOp#cdnn/dense_1/BiasAdd/ReadVariableOp2H
"cdnn/dense_1/MatMul/ReadVariableOp"cdnn/dense_1/MatMul/ReadVariableOp2J
#cdnn/dense_2/BiasAdd/ReadVariableOp#cdnn/dense_2/BiasAdd/ReadVariableOp2H
"cdnn/dense_2/MatMul/ReadVariableOp"cdnn/dense_2/MatMul/ReadVariableOp2J
#cdnn/dense_3/BiasAdd/ReadVariableOp#cdnn/dense_3/BiasAdd/ReadVariableOp2H
"cdnn/dense_3/MatMul/ReadVariableOp"cdnn/dense_3/MatMul/ReadVariableOp2J
#cdnn/dense_4/BiasAdd/ReadVariableOp#cdnn/dense_4/BiasAdd/ReadVariableOp2H
"cdnn/dense_4/MatMul/ReadVariableOp"cdnn/dense_4/MatMul/ReadVariableOp2J
#cdnn/dense_5/BiasAdd/ReadVariableOp#cdnn/dense_5/BiasAdd/ReadVariableOp2H
"cdnn/dense_5/MatMul/ReadVariableOp"cdnn/dense_5/MatMul/ReadVariableOp2J
#cdnn/dense_6/BiasAdd/ReadVariableOp#cdnn/dense_6/BiasAdd/ReadVariableOp2H
"cdnn/dense_6/MatMul/ReadVariableOp"cdnn/dense_6/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
6__inference_batch_normalization_5_layer_call_fn_183254

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_180726o
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
�
b
F__inference_activation_layer_call_and_return_conditional_losses_182697

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

b
C__inference_dropout_layer_call_and_return_conditional_losses_181149

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
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_180562

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
��
�
@__inference_cdnn_layer_call_and_return_conditional_losses_181418
p_tinput
dense_181298:
dense_181300:(
batch_normalization_181303:(
batch_normalization_181305:(
batch_normalization_181307:(
batch_normalization_181309: 
dense_1_181312:
dense_1_181314:*
batch_normalization_1_181317:*
batch_normalization_1_181319:*
batch_normalization_1_181321:*
batch_normalization_1_181323: 
dense_2_181326:
dense_2_181328:*
batch_normalization_2_181331:*
batch_normalization_2_181333:*
batch_normalization_2_181335:*
batch_normalization_2_181337: 
dense_3_181340:
dense_3_181342:*
batch_normalization_3_181345:*
batch_normalization_3_181347:*
batch_normalization_3_181349:*
batch_normalization_3_181351: 
dense_4_181354:
dense_4_181356:*
batch_normalization_4_181359:*
batch_normalization_4_181361:*
batch_normalization_4_181363:*
batch_normalization_4_181365: 
dense_5_181368:
dense_5_181370:*
batch_normalization_5_181373:*
batch_normalization_5_181375:*
batch_normalization_5_181377:*
batch_normalization_5_181379: 
dense_6_181383:
dense_6_181385:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallp_tinputdense_181298dense_181300*
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
A__inference_dense_layer_call_and_return_conditional_losses_180806�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_181303batch_normalization_181305batch_normalization_181307batch_normalization_181309*
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_180363�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_181312dense_1_181314*
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
C__inference_dense_1_layer_call_and_return_conditional_losses_180836�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_181317batch_normalization_1_181319batch_normalization_1_181321batch_normalization_1_181323*
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_180445�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_181326dense_2_181328*
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
C__inference_dense_2_layer_call_and_return_conditional_losses_180866�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_181331batch_normalization_2_181333batch_normalization_2_181335batch_normalization_2_181337*
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_180527�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_181340dense_3_181342*
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
C__inference_dense_3_layer_call_and_return_conditional_losses_180896�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_3_181345batch_normalization_3_181347batch_normalization_3_181349batch_normalization_3_181351*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_180609�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_181354dense_4_181356*
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
C__inference_dense_4_layer_call_and_return_conditional_losses_180926�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_4_181359batch_normalization_4_181361batch_normalization_4_181363batch_normalization_4_181365*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_180691�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_181368dense_5_181370*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_180956�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_5_181373batch_normalization_5_181375batch_normalization_5_181377batch_normalization_5_181379*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_180773�
dropout/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_181149�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_6_181383dense_6_181385*
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
C__inference_dense_6_layer_call_and_return_conditional_losses_180992�
activation/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
F__inference_activation_layer_call_and_return_conditional_losses_181003�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_181298*
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
 *�ŧ7�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_181312*
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
 *�ŧ7�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_181326*
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
 *�ŧ7�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_181340*
_output_shapes

:*
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
 *�ŧ7�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_4_181354*
_output_shapes

:*
dtype0�
&cdnn/dense_4/kernel/Regularizer/L2LossL2Loss=cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_4/kernel/Regularizer/mulMul.cdnn/dense_4/kernel/Regularizer/mul/x:output:0/cdnn/dense_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_5_181368*
_output_shapes

:*
dtype0�
&cdnn/dense_5/kernel/Regularizer/L2LossL2Loss=cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_5/kernel/Regularizer/mulMul.cdnn/dense_5/kernel/Regularizer/mul/x:output:0/cdnn/dense_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_181383*
_output_shapes

:*
dtype0�
&cdnn/dense_6/kernel/Regularizer/L2LossL2Loss=cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_6/kernel/Regularizer/mulMul.cdnn/dense_6/kernel/Regularizer/mul/x:output:0/cdnn/dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tInput
�$
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_180691

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
�
�
6__inference_batch_normalization_3_layer_call_fn_183107

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_180609o
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
(__inference_dense_2_layer_call_fn_182754

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
C__inference_dense_2_layer_call_and_return_conditional_losses_180866o
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
��
�
@__inference_cdnn_layer_call_and_return_conditional_losses_181824
input_1
dense_181704:
dense_181706:(
batch_normalization_181709:(
batch_normalization_181711:(
batch_normalization_181713:(
batch_normalization_181715: 
dense_1_181718:
dense_1_181720:*
batch_normalization_1_181723:*
batch_normalization_1_181725:*
batch_normalization_1_181727:*
batch_normalization_1_181729: 
dense_2_181732:
dense_2_181734:*
batch_normalization_2_181737:*
batch_normalization_2_181739:*
batch_normalization_2_181741:*
batch_normalization_2_181743: 
dense_3_181746:
dense_3_181748:*
batch_normalization_3_181751:*
batch_normalization_3_181753:*
batch_normalization_3_181755:*
batch_normalization_3_181757: 
dense_4_181760:
dense_4_181762:*
batch_normalization_4_181765:*
batch_normalization_4_181767:*
batch_normalization_4_181769:*
batch_normalization_4_181771: 
dense_5_181774:
dense_5_181776:*
batch_normalization_5_181779:*
batch_normalization_5_181781:*
batch_normalization_5_181783:*
batch_normalization_5_181785: 
dense_6_181789:
dense_6_181791:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_181704dense_181706*
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
A__inference_dense_layer_call_and_return_conditional_losses_180806�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_181709batch_normalization_181711batch_normalization_181713batch_normalization_181715*
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_180363�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_181718dense_1_181720*
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
C__inference_dense_1_layer_call_and_return_conditional_losses_180836�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_181723batch_normalization_1_181725batch_normalization_1_181727batch_normalization_1_181729*
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_180445�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_181732dense_2_181734*
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
C__inference_dense_2_layer_call_and_return_conditional_losses_180866�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_181737batch_normalization_2_181739batch_normalization_2_181741batch_normalization_2_181743*
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_180527�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_181746dense_3_181748*
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
C__inference_dense_3_layer_call_and_return_conditional_losses_180896�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_3_181751batch_normalization_3_181753batch_normalization_3_181755batch_normalization_3_181757*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_180609�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_181760dense_4_181762*
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
C__inference_dense_4_layer_call_and_return_conditional_losses_180926�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_4_181765batch_normalization_4_181767batch_normalization_4_181769batch_normalization_4_181771*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_180691�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_181774dense_5_181776*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_180956�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_5_181779batch_normalization_5_181781batch_normalization_5_181783batch_normalization_5_181785*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_180773�
dropout/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_181149�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_6_181789dense_6_181791*
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
C__inference_dense_6_layer_call_and_return_conditional_losses_180992�
activation/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
F__inference_activation_layer_call_and_return_conditional_losses_181003�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_181704*
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
 *�ŧ7�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_181718*
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
 *�ŧ7�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_181732*
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
 *�ŧ7�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_181746*
_output_shapes

:*
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
 *�ŧ7�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_4_181760*
_output_shapes

:*
dtype0�
&cdnn/dense_4/kernel/Regularizer/L2LossL2Loss=cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_4/kernel/Regularizer/mulMul.cdnn/dense_4/kernel/Regularizer/mul/x:output:0/cdnn/dense_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_5_181774*
_output_shapes

:*
dtype0�
&cdnn/dense_5/kernel/Regularizer/L2LossL2Loss=cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_5/kernel/Regularizer/mulMul.cdnn/dense_5/kernel/Regularizer/mul/x:output:0/cdnn/dense_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_181789*
_output_shapes

:*
dtype0�
&cdnn/dense_6/kernel/Regularizer/L2LossL2Loss=cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_6/kernel/Regularizer/mulMul.cdnn/dense_6/kernel/Regularizer/mul/x:output:0/cdnn/dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
6__inference_batch_normalization_2_layer_call_fn_183027

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_180527o
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
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_182652

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
�
�
6__inference_batch_normalization_4_layer_call_fn_183187

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_180691o
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
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183047

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183321

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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_180316

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
�
__inference_loss_fn_3_182586P
>cdnn_dense_3_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_3_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
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
 *�ŧ7�
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
�	
�
__inference_loss_fn_0_182559N
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
 *�ŧ7�
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
�
a
(__inference_dropout_layer_call_fn_182647

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
C__inference_dropout_layer_call_and_return_conditional_losses_181149o
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
�$
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_180445

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
�
�
%__inference_cdnn_layer_call_fn_181578
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

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
 #$%&*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cdnn_layer_call_and_return_conditional_losses_181418o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
A__inference_dense_layer_call_and_return_conditional_losses_180806

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
 *�ŧ7�
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
�$
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183241

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
�
�
%__inference_cdnn_layer_call_fn_182099
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

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallp_tinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
 #$%&*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cdnn_layer_call_and_return_conditional_losses_181418o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�
�
C__inference_dense_6_layer_call_and_return_conditional_losses_180992

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_6/kernel/Regularizer/L2LossL2Loss=cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_6/kernel/Regularizer/mulMul.cdnn/dense_6/kernel/Regularizer/mul/x:output:0/cdnn/dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_3_layer_call_fn_183094

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_180562o
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
(__inference_dense_4_layer_call_fn_182802

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
C__inference_dense_4_layer_call_and_return_conditional_losses_180926o
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
�
�(
@__inference_cdnn_layer_call_and_return_conditional_losses_182546
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
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:K
=batch_normalization_3_assignmovingavg_readvariableop_resource:M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_3_cast_readvariableop_resource:B
4batch_normalization_3_cast_1_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:K
=batch_normalization_4_assignmovingavg_readvariableop_resource:M
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_4_cast_readvariableop_resource:B
4batch_normalization_4_cast_1_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:K
=batch_normalization_5_assignmovingavg_readvariableop_resource:M
?batch_normalization_5_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_5_cast_readvariableop_resource:B
4batch_normalization_5_cast_1_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�%batch_normalization_4/AssignMovingAvg�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_1�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�%batch_normalization_5/AssignMovingAvg�4batch_normalization_5/AssignMovingAvg/ReadVariableOp�'batch_normalization_5/AssignMovingAvg_1�6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_5/Cast/ReadVariableOp�+batch_normalization_5/Cast_1/ReadVariableOp�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�
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
:����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_3/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_3/moments/meanMeandense_3/Relu:activations:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_3/Relu:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:�
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_3/batchnorm/mul_1Muldense_3/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_3/batchnorm/subSub1batch_normalization_3/Cast/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_4/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_4/moments/meanMeandense_4/Relu:activations:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_4/Relu:activations:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:�
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_4/batchnorm/mul_1Muldense_4/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_4/batchnorm/subSub1batch_normalization_4/Cast/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_5/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_5/moments/meanMeandense_5/Relu:activations:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:�
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_5/Relu:activations:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes
:�
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization_5/AssignMovingAvgAssignSubVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
'batch_normalization_5/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_5/Cast/ReadVariableOpReadVariableOp2batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:03batch_normalization_5/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_5/batchnorm/mul_1Muldense_5/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_5/batchnorm/subSub1batch_normalization_5/Cast/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout/dropout/MulMul)batch_normalization_5/batchnorm/add_1:z:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������n
dropout/dropout/ShapeShape)batch_normalization_5/batchnorm/add_1:z:0*
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
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6/MatMulMatMul!dropout/dropout/SelectV2:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
activation/SigmoidSigmoiddense_6/BiasAdd:output:0*
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
 *�ŧ7�
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
 *�ŧ7�
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
 *�ŧ7�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
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
 *�ŧ7�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_4/kernel/Regularizer/L2LossL2Loss=cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_4/kernel/Regularizer/mulMul.cdnn/dense_4/kernel/Regularizer/mul/x:output:0/cdnn/dense_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_5/kernel/Regularizer/L2LossL2Loss=cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_5/kernel/Regularizer/mulMul.cdnn/dense_5/kernel/Regularizer/mul/x:output:0/cdnn/dense_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_6/kernel/Regularizer/L2LossL2Loss=cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_6/kernel/Regularizer/mulMul.cdnn/dense_6/kernel/Regularizer/mul/x:output:0/cdnn/dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2N
%batch_normalization_5/AssignMovingAvg%batch_normalization_5/AssignMovingAvg2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_1'batch_normalization_5/AssignMovingAvg_12p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_5/Cast/ReadVariableOp)batch_normalization_5/Cast/ReadVariableOp2Z
+batch_normalization_5/Cast_1/ReadVariableOp+batch_normalization_5/Cast_1/ReadVariableOp2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�
�
(__inference_dense_1_layer_call_fn_182730

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
C__inference_dense_1_layer_call_and_return_conditional_losses_180836o
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
C__inference_dense_4_layer_call_and_return_conditional_losses_182817

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_4/kernel/Regularizer/L2LossL2Loss=cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_4/kernel/Regularizer/mulMul.cdnn/dense_4/kernel/Regularizer/mul/x:output:0/cdnn/dense_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_182921

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
�
�
%__inference_cdnn_layer_call_fn_182018
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

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallp_tinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cdnn_layer_call_and_return_conditional_losses_181034o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�$
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183001

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
�
�
C__inference_dense_5_layer_call_and_return_conditional_losses_182841

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_5/kernel/Regularizer/L2LossL2Loss=cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_5/kernel/Regularizer/mulMul.cdnn/dense_5/kernel/Regularizer/mul/x:output:0/cdnn/dense_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
@__inference_cdnn_layer_call_and_return_conditional_losses_181034
p_tinput
dense_180807:
dense_180809:(
batch_normalization_180812:(
batch_normalization_180814:(
batch_normalization_180816:(
batch_normalization_180818: 
dense_1_180837:
dense_1_180839:*
batch_normalization_1_180842:*
batch_normalization_1_180844:*
batch_normalization_1_180846:*
batch_normalization_1_180848: 
dense_2_180867:
dense_2_180869:*
batch_normalization_2_180872:*
batch_normalization_2_180874:*
batch_normalization_2_180876:*
batch_normalization_2_180878: 
dense_3_180897:
dense_3_180899:*
batch_normalization_3_180902:*
batch_normalization_3_180904:*
batch_normalization_3_180906:*
batch_normalization_3_180908: 
dense_4_180927:
dense_4_180929:*
batch_normalization_4_180932:*
batch_normalization_4_180934:*
batch_normalization_4_180936:*
batch_normalization_4_180938: 
dense_5_180957:
dense_5_180959:*
batch_normalization_5_180962:*
batch_normalization_5_180964:*
batch_normalization_5_180966:*
batch_normalization_5_180968: 
dense_6_180993:
dense_6_180995:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallp_tinputdense_180807dense_180809*
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
A__inference_dense_layer_call_and_return_conditional_losses_180806�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_180812batch_normalization_180814batch_normalization_180816batch_normalization_180818*
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_180316�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_180837dense_1_180839*
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
C__inference_dense_1_layer_call_and_return_conditional_losses_180836�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_180842batch_normalization_1_180844batch_normalization_1_180846batch_normalization_1_180848*
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_180398�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_180867dense_2_180869*
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
C__inference_dense_2_layer_call_and_return_conditional_losses_180866�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_180872batch_normalization_2_180874batch_normalization_2_180876batch_normalization_2_180878*
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_180480�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_180897dense_3_180899*
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
C__inference_dense_3_layer_call_and_return_conditional_losses_180896�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_3_180902batch_normalization_3_180904batch_normalization_3_180906batch_normalization_3_180908*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_180562�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_180927dense_4_180929*
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
C__inference_dense_4_layer_call_and_return_conditional_losses_180926�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_4_180932batch_normalization_4_180934batch_normalization_4_180936batch_normalization_4_180938*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_180644�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_180957dense_5_180959*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_180956�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_5_180962batch_normalization_5_180964batch_normalization_5_180966batch_normalization_5_180968*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_180726�
dropout/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_180976�
dense_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_6_180993dense_6_180995*
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
C__inference_dense_6_layer_call_and_return_conditional_losses_180992�
activation/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
F__inference_activation_layer_call_and_return_conditional_losses_181003�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_180807*
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
 *�ŧ7�
!cdnn/dense/kernel/Regularizer/mulMul,cdnn/dense/kernel/Regularizer/mul/x:output:0-cdnn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_180837*
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
 *�ŧ7�
#cdnn/dense_1/kernel/Regularizer/mulMul.cdnn/dense_1/kernel/Regularizer/mul/x:output:0/cdnn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_180867*
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
 *�ŧ7�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_180897*
_output_shapes

:*
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
 *�ŧ7�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_4_180927*
_output_shapes

:*
dtype0�
&cdnn/dense_4/kernel/Regularizer/L2LossL2Loss=cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_4/kernel/Regularizer/mulMul.cdnn/dense_4/kernel/Regularizer/mul/x:output:0/cdnn/dense_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_5_180957*
_output_shapes

:*
dtype0�
&cdnn/dense_5/kernel/Regularizer/L2LossL2Loss=cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_5/kernel/Regularizer/mulMul.cdnn/dense_5/kernel/Regularizer/mul/x:output:0/cdnn/dense_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_180993*
_output_shapes

:*
dtype0�
&cdnn/dense_6/kernel/Regularizer/L2LossL2Loss=cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_6/kernel/Regularizer/mulMul.cdnn/dense_6/kernel/Regularizer/mul/x:output:0/cdnn/dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tInput
�
�
4__inference_batch_normalization_layer_call_fn_182867

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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_180363o
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
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_182769

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
 *�ŧ7�
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
�
__inference_loss_fn_1_182568P
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
 *�ŧ7�
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
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_182664

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
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183127

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
�
�
$__inference_signature_wrapper_181937
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

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_180292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_180480

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
��
�"
@__inference_cdnn_layer_call_and_return_conditional_losses_182277
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
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:@
2batch_normalization_3_cast_readvariableop_resource:B
4batch_normalization_3_cast_1_readvariableop_resource:B
4batch_normalization_3_cast_2_readvariableop_resource:B
4batch_normalization_3_cast_3_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:@
2batch_normalization_4_cast_readvariableop_resource:B
4batch_normalization_4_cast_1_readvariableop_resource:B
4batch_normalization_4_cast_2_readvariableop_resource:B
4batch_normalization_4_cast_3_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:@
2batch_normalization_5_cast_readvariableop_resource:B
4batch_normalization_5_cast_1_readvariableop_resource:B
4batch_normalization_5_cast_2_readvariableop_resource:B
4batch_normalization_5_cast_3_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity��'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�)batch_normalization/Cast_2/ReadVariableOp�)batch_normalization/Cast_3/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�+batch_normalization_1/Cast_2/ReadVariableOp�+batch_normalization_1/Cast_3/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�+batch_normalization_2/Cast_2/ReadVariableOp�+batch_normalization_2/Cast_3/ReadVariableOp�)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�+batch_normalization_3/Cast_2/ReadVariableOp�+batch_normalization_3/Cast_3/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�+batch_normalization_4/Cast_2/ReadVariableOp�+batch_normalization_4/Cast_3/ReadVariableOp�)batch_normalization_5/Cast/ReadVariableOp�+batch_normalization_5/Cast_1/ReadVariableOp�+batch_normalization_5/Cast_2/ReadVariableOp�+batch_normalization_5/Cast_3/ReadVariableOp�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�
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
:����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_3/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:����������
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_3/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_3/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_3/batchnorm/addAddV23batch_normalization_3/Cast_1/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_3/batchnorm/mul_1Muldense_3/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_3/batchnorm/mul_2Mul1batch_normalization_3/Cast/ReadVariableOp:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_3/batchnorm/subSub3batch_normalization_3/Cast_2/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_4/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV23batch_normalization_4/Cast_1/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_4/batchnorm/mul_1Muldense_4/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_4/batchnorm/mul_2Mul1batch_normalization_4/Cast/ReadVariableOp:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_4/batchnorm/subSub3batch_normalization_4/Cast_2/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_5/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:����������
)batch_normalization_5/Cast/ReadVariableOpReadVariableOp2batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_5/batchnorm/addAddV23batch_normalization_5/Cast_1/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:03batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_5/batchnorm/mul_1Muldense_5/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_5/batchnorm/mul_2Mul1batch_normalization_5/Cast/ReadVariableOp:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_5/batchnorm/subSub3batch_normalization_5/Cast_2/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������y
dropout/IdentityIdentity)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6/MatMulMatMuldropout/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
activation/SigmoidSigmoiddense_6/BiasAdd:output:0*
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
 *�ŧ7�
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
 *�ŧ7�
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
 *�ŧ7�
#cdnn/dense_2/kernel/Regularizer/mulMul.cdnn/dense_2/kernel/Regularizer/mul/x:output:0/cdnn/dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
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
 *�ŧ7�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_4/kernel/Regularizer/L2LossL2Loss=cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_4/kernel/Regularizer/mulMul.cdnn/dense_4/kernel/Regularizer/mul/x:output:0/cdnn/dense_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_5/kernel/Regularizer/L2LossL2Loss=cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_5/kernel/Regularizer/mulMul.cdnn/dense_5/kernel/Regularizer/mul/x:output:0/cdnn/dense_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_6/kernel/Regularizer/L2LossL2Loss=cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_6/kernel/Regularizer/mulMul.cdnn/dense_6/kernel/Regularizer/mul/x:output:0/cdnn/dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp,^batch_normalization_3/Cast_2/ReadVariableOp,^batch_normalization_3/Cast_3/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp,^batch_normalization_5/Cast_2/ReadVariableOp,^batch_normalization_5/Cast_3/ReadVariableOp4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
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
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2Z
+batch_normalization_3/Cast_2/ReadVariableOp+batch_normalization_3/Cast_2/ReadVariableOp2Z
+batch_normalization_3/Cast_3/ReadVariableOp+batch_normalization_3/Cast_3/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2Z
+batch_normalization_4/Cast_2/ReadVariableOp+batch_normalization_4/Cast_2/ReadVariableOp2Z
+batch_normalization_4/Cast_3/ReadVariableOp+batch_normalization_4/Cast_3/ReadVariableOp2V
)batch_normalization_5/Cast/ReadVariableOp)batch_normalization_5/Cast/ReadVariableOp2Z
+batch_normalization_5/Cast_1/ReadVariableOp+batch_normalization_5/Cast_1/ReadVariableOp2Z
+batch_normalization_5/Cast_2/ReadVariableOp+batch_normalization_5/Cast_2/ReadVariableOp2Z
+batch_normalization_5/Cast_3/ReadVariableOp+batch_normalization_5/Cast_3/ReadVariableOp2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�	
�
__inference_loss_fn_5_182604P
>cdnn_dense_5_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_5_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_5/kernel/Regularizer/L2LossL2Loss=cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_5/kernel/Regularizer/mulMul.cdnn/dense_5/kernel/Regularizer/mul/x:output:0/cdnn/dense_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'cdnn/dense_5/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
6__inference_batch_normalization_5_layer_call_fn_183267

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_180773o
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
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183207

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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_180363

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
�
�
6__inference_batch_normalization_1_layer_call_fn_182934

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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_180398o
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
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_180644

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
�
�
C__inference_dense_5_layer_call_and_return_conditional_losses_180956

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_5/kernel/Regularizer/L2LossL2Loss=cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_5/kernel/Regularizer/mulMul.cdnn/dense_5/kernel/Regularizer/mul/x:output:0/cdnn/dense_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_182745

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
 *�ŧ7�
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
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_180896

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
 *�ŧ7�
#cdnn/dense_3/kernel/Regularizer/mulMul.cdnn/dense_3/kernel/Regularizer/mul/x:output:0/cdnn/dense_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
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
�
�
(__inference_dense_3_layer_call_fn_182778

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
C__inference_dense_3_layer_call_and_return_conditional_losses_180896o
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
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_182887

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
a
C__inference_dropout_layer_call_and_return_conditional_losses_180976

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
C__inference_dense_2_layer_call_and_return_conditional_losses_180866

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
 *�ŧ7�
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
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_180398

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
�
�
C__inference_dense_6_layer_call_and_return_conditional_losses_182687

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_6/kernel/Regularizer/L2LossL2Loss=cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_6/kernel/Regularizer/mulMul.cdnn/dense_6/kernel/Regularizer/mul/x:output:0/cdnn/dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_cdnn_layer_call_fn_181113
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

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_cdnn_layer_call_and_return_conditional_losses_181034o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
(__inference_dense_5_layer_call_fn_182826

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
C__inference_dense_5_layer_call_and_return_conditional_losses_180956o
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
C__inference_dense_1_layer_call_and_return_conditional_losses_180836

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
 *�ŧ7�
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
�$
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183081

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
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
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
�
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
$19
%20
&21
'22
(23
)24
*25
+26
,27
-28
.29
/30
031
132
233
334
435
536
637"
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
12
13
14
 15
!16
"17
#18
$19
%20
&21
'22
(23
524
625"
trackable_list_wrapper
Q
70
81
92
:3
;4
<5
=6"
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32�
%__inference_cdnn_layer_call_fn_181113
%__inference_cdnn_layer_call_fn_182018
%__inference_cdnn_layer_call_fn_182099
%__inference_cdnn_layer_call_fn_181578�
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
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
�
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32�
@__inference_cdnn_layer_call_and_return_conditional_losses_182277
@__inference_cdnn_layer_call_and_return_conditional_losses_182546
@__inference_cdnn_layer_call_and_return_conditional_losses_181701
@__inference_cdnn_layer_call_and_return_conditional_losses_181824�
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
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
�B�
!__inference__wrapped_model_180292input_1"�
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
KDNN.LayerNeurons"
trackable_dict_wrapper
 "
trackable_list_wrapper
J
L0
M1
N2
O3
P4
Q5"
trackable_list_wrapper
J
R0
S1
T2
U3
V4
W5"
trackable_list_wrapper
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
�
k
_variables
l_iterations
m_learning_rate
n_index_dict
o	momentums
p_update_step_xla"
experimentalOptimizer
,
qserving_default"
signature_map
#:!2cdnn/dense/kernel
:2cdnn/dense/bias
%:#2cdnn/dense_1/kernel
:2cdnn/dense_1/bias
%:#2cdnn/dense_2/kernel
:2cdnn/dense_2/bias
%:#2cdnn/dense_3/kernel
:2cdnn/dense_3/bias
%:#2cdnn/dense_4/kernel
:2cdnn/dense_4/bias
%:#2cdnn/dense_5/kernel
:2cdnn/dense_5/bias
,:*2cdnn/batch_normalization/gamma
+:)2cdnn/batch_normalization/beta
.:,2 cdnn/batch_normalization_1/gamma
-:+2cdnn/batch_normalization_1/beta
.:,2 cdnn/batch_normalization_2/gamma
-:+2cdnn/batch_normalization_2/beta
.:,2 cdnn/batch_normalization_3/gamma
-:+2cdnn/batch_normalization_3/beta
.:,2 cdnn/batch_normalization_4/gamma
-:+2cdnn/batch_normalization_4/beta
.:,2 cdnn/batch_normalization_5/gamma
-:+2cdnn/batch_normalization_5/beta
4:2 (2$cdnn/batch_normalization/moving_mean
8:6 (2(cdnn/batch_normalization/moving_variance
6:4 (2&cdnn/batch_normalization_1/moving_mean
::8 (2*cdnn/batch_normalization_1/moving_variance
6:4 (2&cdnn/batch_normalization_2/moving_mean
::8 (2*cdnn/batch_normalization_2/moving_variance
6:4 (2&cdnn/batch_normalization_3/moving_mean
::8 (2*cdnn/batch_normalization_3/moving_variance
6:4 (2&cdnn/batch_normalization_4/moving_mean
::8 (2*cdnn/batch_normalization_4/moving_variance
6:4 (2&cdnn/batch_normalization_5/moving_mean
::8 (2*cdnn/batch_normalization_5/moving_variance
%:#2cdnn/dense_6/kernel
:2cdnn/dense_6/bias
�
rtrace_02�
__inference_loss_fn_0_182559�
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
annotations� *� zrtrace_0
�
strace_02�
__inference_loss_fn_1_182568�
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
annotations� *� zstrace_0
�
ttrace_02�
__inference_loss_fn_2_182577�
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
annotations� *� zttrace_0
�
utrace_02�
__inference_loss_fn_3_182586�
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
annotations� *� zutrace_0
�
vtrace_02�
__inference_loss_fn_4_182595�
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
annotations� *� zvtrace_0
�
wtrace_02�
__inference_loss_fn_5_182604�
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
annotations� *� zwtrace_0
�
xtrace_02�
__inference_loss_fn_6_182613�
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
annotations� *� zxtrace_0
v
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411"
trackable_list_wrapper
�
L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9
V10
W11
12
13
14"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_cdnn_layer_call_fn_181113input_1"�
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
%__inference_cdnn_layer_call_fn_182018p_tinput"�
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
%__inference_cdnn_layer_call_fn_182099p_tinput"�
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
%__inference_cdnn_layer_call_fn_181578input_1"�
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
@__inference_cdnn_layer_call_and_return_conditional_losses_182277p_tinput"�
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
@__inference_cdnn_layer_call_and_return_conditional_losses_182546p_tinput"�
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
@__inference_cdnn_layer_call_and_return_conditional_losses_181701input_1"�
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
@__inference_cdnn_layer_call_and_return_conditional_losses_181824input_1"�
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
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
)moving_mean
*moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
 beta
+moving_mean
,moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	!gamma
"beta
-moving_mean
.moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	#gamma
$beta
/moving_mean
0moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	%gamma
&beta
1moving_mean
2moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	'gamma
(beta
3moving_mean
4moving_variance"
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
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_layer_call_fn_182642
(__inference_dropout_layer_call_fn_182647�
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
C__inference_dropout_layer_call_and_return_conditional_losses_182652
C__inference_dropout_layer_call_and_return_conditional_losses_182664�
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
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_6_layer_call_fn_182673�
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
C__inference_dense_6_layer_call_and_return_conditional_losses_182687�
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
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_activation_layer_call_fn_182692�
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
F__inference_activation_layer_call_and_return_conditional_losses_182697�
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
l0
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
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
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
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25"
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
$__inference_signature_wrapper_181937input_1"�
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
__inference_loss_fn_0_182559"�
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
__inference_loss_fn_1_182568"�
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
__inference_loss_fn_2_182577"�
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
__inference_loss_fn_3_182586"�
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
__inference_loss_fn_4_182595"�
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
__inference_loss_fn_5_182604"�
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
__inference_loss_fn_6_182613"�
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

�total

�count
�
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
70"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_182706�
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
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_182721�
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
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
80"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_182730�
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
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_182745�
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
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
90"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_182754�
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
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_182769�
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
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_3_layer_call_fn_182778�
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
 z�trace_0
�
�trace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_182793�
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
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
;0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_4_layer_call_fn_182802�
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
 z�trace_0
�
�trace_02�
C__inference_dense_4_layer_call_and_return_conditional_losses_182817�
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
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_5_layer_call_fn_182826�
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
 z�trace_0
�
�trace_02�
C__inference_dense_5_layer_call_and_return_conditional_losses_182841�
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
 z�trace_0
<
0
1
)2
*3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_layer_call_fn_182854
4__inference_batch_normalization_layer_call_fn_182867�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_182887
O__inference_batch_normalization_layer_call_and_return_conditional_losses_182921�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
0
 1
+2
,3"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_1_layer_call_fn_182934
6__inference_batch_normalization_1_layer_call_fn_182947�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182967
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183001�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
!0
"1
-2
.3"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_2_layer_call_fn_183014
6__inference_batch_normalization_2_layer_call_fn_183027�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183047
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183081�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
#0
$1
/2
03"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_3_layer_call_fn_183094
6__inference_batch_normalization_3_layer_call_fn_183107�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183127
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183161�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
%0
&1
12
23"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_4_layer_call_fn_183174
6__inference_batch_normalization_4_layer_call_fn_183187�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183207
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183241�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
'0
(1
32
43"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_5_layer_call_fn_183254
6__inference_batch_normalization_5_layer_call_fn_183267�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183287
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183321�
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
 z�trace_0z�trace_1
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
(__inference_dropout_layer_call_fn_182642inputs"�
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
(__inference_dropout_layer_call_fn_182647inputs"�
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
C__inference_dropout_layer_call_and_return_conditional_losses_182652inputs"�
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
C__inference_dropout_layer_call_and_return_conditional_losses_182664inputs"�
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
=0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_6_layer_call_fn_182673inputs"�
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
C__inference_dense_6_layer_call_and_return_conditional_losses_182687inputs"�
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
+__inference_activation_layer_call_fn_182692inputs"�
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
F__inference_activation_layer_call_and_return_conditional_losses_182697inputs"�
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
):'2SGD/m/cdnn/dense_3/kernel
#:!2SGD/m/cdnn/dense_3/bias
):'2SGD/m/cdnn/dense_4/kernel
#:!2SGD/m/cdnn/dense_4/bias
):'2SGD/m/cdnn/dense_5/kernel
#:!2SGD/m/cdnn/dense_5/bias
0:.2$SGD/m/cdnn/batch_normalization/gamma
/:-2#SGD/m/cdnn/batch_normalization/beta
2:02&SGD/m/cdnn/batch_normalization_1/gamma
1:/2%SGD/m/cdnn/batch_normalization_1/beta
2:02&SGD/m/cdnn/batch_normalization_2/gamma
1:/2%SGD/m/cdnn/batch_normalization_2/beta
2:02&SGD/m/cdnn/batch_normalization_3/gamma
1:/2%SGD/m/cdnn/batch_normalization_3/beta
2:02&SGD/m/cdnn/batch_normalization_4/gamma
1:/2%SGD/m/cdnn/batch_normalization_4/beta
2:02&SGD/m/cdnn/batch_normalization_5/gamma
1:/2%SGD/m/cdnn/batch_normalization_5/beta
):'2SGD/m/cdnn/dense_6/kernel
#:!2SGD/m/cdnn/dense_6/bias
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
�0
�1"
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
70"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_layer_call_fn_182706inputs"�
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
A__inference_dense_layer_call_and_return_conditional_losses_182721inputs"�
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
80"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_1_layer_call_fn_182730inputs"�
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
C__inference_dense_1_layer_call_and_return_conditional_losses_182745inputs"�
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
90"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_182754inputs"�
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
C__inference_dense_2_layer_call_and_return_conditional_losses_182769inputs"�
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
:0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_3_layer_call_fn_182778inputs"�
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
C__inference_dense_3_layer_call_and_return_conditional_losses_182793inputs"�
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
;0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_4_layer_call_fn_182802inputs"�
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
C__inference_dense_4_layer_call_and_return_conditional_losses_182817inputs"�
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
<0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_5_layer_call_fn_182826inputs"�
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
C__inference_dense_5_layer_call_and_return_conditional_losses_182841inputs"�
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
)0
*1"
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
4__inference_batch_normalization_layer_call_fn_182854inputs"�
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
4__inference_batch_normalization_layer_call_fn_182867inputs"�
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_182887inputs"�
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_182921inputs"�
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
+0
,1"
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
6__inference_batch_normalization_1_layer_call_fn_182934inputs"�
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
6__inference_batch_normalization_1_layer_call_fn_182947inputs"�
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182967inputs"�
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183001inputs"�
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
-0
.1"
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
6__inference_batch_normalization_2_layer_call_fn_183014inputs"�
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
6__inference_batch_normalization_2_layer_call_fn_183027inputs"�
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183047inputs"�
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183081inputs"�
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
/0
01"
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
6__inference_batch_normalization_3_layer_call_fn_183094inputs"�
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
6__inference_batch_normalization_3_layer_call_fn_183107inputs"�
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183127inputs"�
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183161inputs"�
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
10
21"
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
6__inference_batch_normalization_4_layer_call_fn_183174inputs"�
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
6__inference_batch_normalization_4_layer_call_fn_183187inputs"�
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183207inputs"�
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183241inputs"�
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
30
41"
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
6__inference_batch_normalization_5_layer_call_fn_183254inputs"�
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
6__inference_batch_normalization_5_layer_call_fn_183267inputs"�
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183287inputs"�
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183321inputs"�
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
!__inference__wrapped_model_180292�&)*+, -."!/0$#12&%34('560�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
F__inference_activation_layer_call_and_return_conditional_losses_182697_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
+__inference_activation_layer_call_fn_182692T/�,
%�"
 �
inputs���������
� "!�
unknown����������
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182967i+, 3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183001i+, 3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
6__inference_batch_normalization_1_layer_call_fn_182934^+, 3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
6__inference_batch_normalization_1_layer_call_fn_182947^+, 3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183047i-."!3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183081i-."!3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
6__inference_batch_normalization_2_layer_call_fn_183014^-."!3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
6__inference_batch_normalization_2_layer_call_fn_183027^-."!3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183127i/0$#3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183161i/0$#3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
6__inference_batch_normalization_3_layer_call_fn_183094^/0$#3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
6__inference_batch_normalization_3_layer_call_fn_183107^/0$#3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183207i12&%3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183241i12&%3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
6__inference_batch_normalization_4_layer_call_fn_183174^12&%3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
6__inference_batch_normalization_4_layer_call_fn_183187^12&%3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183287i34('3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183321i34('3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
6__inference_batch_normalization_5_layer_call_fn_183254^34('3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
6__inference_batch_normalization_5_layer_call_fn_183267^34('3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
O__inference_batch_normalization_layer_call_and_return_conditional_losses_182887i)*3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_182921i)*3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
4__inference_batch_normalization_layer_call_fn_182854^)*3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
4__inference_batch_normalization_layer_call_fn_182867^)*3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
@__inference_cdnn_layer_call_and_return_conditional_losses_181701�&)*+, -."!/0$#12&%34('56@�=
&�#
!�
input_1���������
�

trainingp ",�)
"�
tensor_0���������
� �
@__inference_cdnn_layer_call_and_return_conditional_losses_181824�&)*+, -."!/0$#12&%34('56@�=
&�#
!�
input_1���������
�

trainingp",�)
"�
tensor_0���������
� �
@__inference_cdnn_layer_call_and_return_conditional_losses_182277�&)*+, -."!/0$#12&%34('56A�>
'�$
"�
p_tinput���������
�

trainingp ",�)
"�
tensor_0���������
� �
@__inference_cdnn_layer_call_and_return_conditional_losses_182546�&)*+, -."!/0$#12&%34('56A�>
'�$
"�
p_tinput���������
�

trainingp",�)
"�
tensor_0���������
� �
%__inference_cdnn_layer_call_fn_181113�&)*+, -."!/0$#12&%34('56@�=
&�#
!�
input_1���������
�

trainingp "!�
unknown����������
%__inference_cdnn_layer_call_fn_181578�&)*+, -."!/0$#12&%34('56@�=
&�#
!�
input_1���������
�

trainingp"!�
unknown����������
%__inference_cdnn_layer_call_fn_182018�&)*+, -."!/0$#12&%34('56A�>
'�$
"�
p_tinput���������
�

trainingp "!�
unknown����������
%__inference_cdnn_layer_call_fn_182099�&)*+, -."!/0$#12&%34('56A�>
'�$
"�
p_tinput���������
�

trainingp"!�
unknown����������
C__inference_dense_1_layer_call_and_return_conditional_losses_182745c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_1_layer_call_fn_182730X/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dense_2_layer_call_and_return_conditional_losses_182769c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_2_layer_call_fn_182754X/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dense_3_layer_call_and_return_conditional_losses_182793c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_3_layer_call_fn_182778X/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dense_4_layer_call_and_return_conditional_losses_182817c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_4_layer_call_fn_182802X/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dense_5_layer_call_and_return_conditional_losses_182841c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_5_layer_call_fn_182826X/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dense_6_layer_call_and_return_conditional_losses_182687c56/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_6_layer_call_fn_182673X56/�,
%�"
 �
inputs���������
� "!�
unknown����������
A__inference_dense_layer_call_and_return_conditional_losses_182721c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
&__inference_dense_layer_call_fn_182706X/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dropout_layer_call_and_return_conditional_losses_182652c3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
C__inference_dropout_layer_call_and_return_conditional_losses_182664c3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
(__inference_dropout_layer_call_fn_182642X3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
(__inference_dropout_layer_call_fn_182647X3�0
)�&
 �
inputs���������
p
� "!�
unknown���������D
__inference_loss_fn_0_182559$�

� 
� "�
unknown D
__inference_loss_fn_1_182568$�

� 
� "�
unknown D
__inference_loss_fn_2_182577$�

� 
� "�
unknown D
__inference_loss_fn_3_182586$�

� 
� "�
unknown D
__inference_loss_fn_4_182595$�

� 
� "�
unknown D
__inference_loss_fn_5_182604$�

� 
� "�
unknown D
__inference_loss_fn_6_182613$5�

� 
� "�
unknown �
$__inference_signature_wrapper_181937�&)*+, -."!/0$#12&%34('56;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������