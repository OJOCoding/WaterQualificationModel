��+
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
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��&
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
SGD/m/cdnn/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_9/bias

+SGD/m/cdnn/dense_9/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_9/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_9/kernel
�
-SGD/m/cdnn/dense_9/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_9/kernel*
_output_shapes

:*
dtype0
�
%SGD/m/cdnn/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/m/cdnn/batch_normalization_8/beta
�
9SGD/m/cdnn/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp%SGD/m/cdnn/batch_normalization_8/beta*
_output_shapes
:*
dtype0
�
&SGD/m/cdnn/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/m/cdnn/batch_normalization_8/gamma
�
:SGD/m/cdnn/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp&SGD/m/cdnn/batch_normalization_8/gamma*
_output_shapes
:*
dtype0
�
%SGD/m/cdnn/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/m/cdnn/batch_normalization_7/beta
�
9SGD/m/cdnn/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp%SGD/m/cdnn/batch_normalization_7/beta*
_output_shapes
:*
dtype0
�
&SGD/m/cdnn/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/m/cdnn/batch_normalization_7/gamma
�
:SGD/m/cdnn/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp&SGD/m/cdnn/batch_normalization_7/gamma*
_output_shapes
:*
dtype0
�
%SGD/m/cdnn/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%SGD/m/cdnn/batch_normalization_6/beta
�
9SGD/m/cdnn/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp%SGD/m/cdnn/batch_normalization_6/beta*
_output_shapes
:*
dtype0
�
&SGD/m/cdnn/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/m/cdnn/batch_normalization_6/gamma
�
:SGD/m/cdnn/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp&SGD/m/cdnn/batch_normalization_6/gamma*
_output_shapes
:*
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
SGD/m/cdnn/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_8/bias

+SGD/m/cdnn/dense_8/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_8/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_8/kernel
�
-SGD/m/cdnn/dense_8/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_8/kernel*
_output_shapes

:*
dtype0
�
SGD/m/cdnn/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_7/bias

+SGD/m/cdnn/dense_7/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_7/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_7/kernel
�
-SGD/m/cdnn/dense_7/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_7/kernel*
_output_shapes

:*
dtype0
�
SGD/m/cdnn/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/cdnn/dense_6/bias

+SGD/m/cdnn/dense_6/bias/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_6/bias*
_output_shapes
:*
dtype0
�
SGD/m/cdnn/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameSGD/m/cdnn/dense_6/kernel
�
-SGD/m/cdnn/dense_6/kernel/Read/ReadVariableOpReadVariableOpSGD/m/cdnn/dense_6/kernel*
_output_shapes

:*
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
cdnn/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_9/bias
s
%cdnn/dense_9/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_9/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_9/kernel
{
'cdnn/dense_9/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_9/kernel*
_output_shapes

:*
dtype0
�
*cdnn/batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*cdnn/batch_normalization_8/moving_variance
�
>cdnn/batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp*cdnn/batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0
�
&cdnn/batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&cdnn/batch_normalization_8/moving_mean
�
:cdnn/batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp&cdnn/batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
�
*cdnn/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*cdnn/batch_normalization_7/moving_variance
�
>cdnn/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp*cdnn/batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0
�
&cdnn/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&cdnn/batch_normalization_7/moving_mean
�
:cdnn/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp&cdnn/batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
�
*cdnn/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*cdnn/batch_normalization_6/moving_variance
�
>cdnn/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp*cdnn/batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0
�
&cdnn/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&cdnn/batch_normalization_6/moving_mean
�
:cdnn/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp&cdnn/batch_normalization_6/moving_mean*
_output_shapes
:*
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
cdnn/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cdnn/batch_normalization_8/beta
�
3cdnn/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization_8/beta*
_output_shapes
:*
dtype0
�
 cdnn/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cdnn/batch_normalization_8/gamma
�
4cdnn/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp cdnn/batch_normalization_8/gamma*
_output_shapes
:*
dtype0
�
cdnn/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cdnn/batch_normalization_7/beta
�
3cdnn/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization_7/beta*
_output_shapes
:*
dtype0
�
 cdnn/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cdnn/batch_normalization_7/gamma
�
4cdnn/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp cdnn/batch_normalization_7/gamma*
_output_shapes
:*
dtype0
�
cdnn/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cdnn/batch_normalization_6/beta
�
3cdnn/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpcdnn/batch_normalization_6/beta*
_output_shapes
:*
dtype0
�
 cdnn/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cdnn/batch_normalization_6/gamma
�
4cdnn/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp cdnn/batch_normalization_6/gamma*
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
cdnn/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_8/bias
s
%cdnn/dense_8/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_8/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_8/kernel
{
'cdnn/dense_8/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_8/kernel*
_output_shapes

:*
dtype0
z
cdnn/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_7/bias
s
%cdnn/dense_7/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_7/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_7/kernel
{
'cdnn/dense_7/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_7/kernel*
_output_shapes

:*
dtype0
z
cdnn/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namecdnn/dense_6/bias
s
%cdnn/dense_6/bias/Read/ReadVariableOpReadVariableOpcdnn/dense_6/bias*
_output_shapes
:*
dtype0
�
cdnn/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namecdnn/dense_6/kernel
{
'cdnn/dense_6/kernel/Read/ReadVariableOpReadVariableOpcdnn/dense_6/kernel*
_output_shapes

:*
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
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cdnn/dense/kernelcdnn/dense/bias$cdnn/batch_normalization/moving_mean(cdnn/batch_normalization/moving_variancecdnn/batch_normalization/betacdnn/batch_normalization/gammacdnn/dense_1/kernelcdnn/dense_1/bias&cdnn/batch_normalization_1/moving_mean*cdnn/batch_normalization_1/moving_variancecdnn/batch_normalization_1/beta cdnn/batch_normalization_1/gammacdnn/dense_2/kernelcdnn/dense_2/bias&cdnn/batch_normalization_2/moving_mean*cdnn/batch_normalization_2/moving_variancecdnn/batch_normalization_2/beta cdnn/batch_normalization_2/gammacdnn/dense_3/kernelcdnn/dense_3/bias&cdnn/batch_normalization_3/moving_mean*cdnn/batch_normalization_3/moving_variancecdnn/batch_normalization_3/beta cdnn/batch_normalization_3/gammacdnn/dense_4/kernelcdnn/dense_4/bias&cdnn/batch_normalization_4/moving_mean*cdnn/batch_normalization_4/moving_variancecdnn/batch_normalization_4/beta cdnn/batch_normalization_4/gammacdnn/dense_5/kernelcdnn/dense_5/bias&cdnn/batch_normalization_5/moving_mean*cdnn/batch_normalization_5/moving_variancecdnn/batch_normalization_5/beta cdnn/batch_normalization_5/gammacdnn/dense_6/kernelcdnn/dense_6/bias&cdnn/batch_normalization_6/moving_mean*cdnn/batch_normalization_6/moving_variancecdnn/batch_normalization_6/beta cdnn/batch_normalization_6/gammacdnn/dense_7/kernelcdnn/dense_7/bias&cdnn/batch_normalization_7/moving_mean*cdnn/batch_normalization_7/moving_variancecdnn/batch_normalization_7/beta cdnn/batch_normalization_7/gammacdnn/dense_8/kernelcdnn/dense_8/bias&cdnn/batch_normalization_8/moving_mean*cdnn/batch_normalization_8/moving_variancecdnn/batch_normalization_8/beta cdnn/batch_normalization_8/gammacdnn/dense_9/kernelcdnn/dense_9/bias*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1090103

NoOpNoOp
��
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
�
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
637
738
839
940
:41
;42
<43
=44
>45
?46
@47
A48
B49
C50
D51
E52
F53
G54
H55*
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
G36
H37*
H
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9* 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3* 
6
\trace_0
]trace_1
^trace_2
_trace_3* 
* 

`DNN.LayerNeurons* 
* 
C
a0
b1
c2
d3
e4
f5
g6
h7
i8*
C
j0
k1
l2
m3
n4
o5
p6
q7
r8*
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
y_random_generator* 
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

Gkernel
Hbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
u
�
_variables
�_iterations
�_learning_rate
�_index_dict
�	momentums
�_update_step_xla*

�serving_default* 
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
TN
VARIABLE_VALUEcdnn/dense_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEcdnn/dense_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcdnn/dense_7/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEcdnn/dense_7/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcdnn/dense_8/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEcdnn/dense_8/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEcdnn/batch_normalization/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcdnn/batch_normalization/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_1/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_1/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_2/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_2/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_3/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_3/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_4/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_4/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_5/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_5/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_6/gamma'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_6/beta'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_7/gamma'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_7/beta'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE cdnn/batch_normalization_8/gamma'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEcdnn/batch_normalization_8/beta'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$cdnn/batch_normalization/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(cdnn/batch_normalization/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_1/moving_mean'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_1/moving_variance'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_2/moving_mean'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_2/moving_variance'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_3/moving_mean'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_3/moving_variance'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_4/moving_mean'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_4/moving_variance'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_5/moving_mean'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_5/moving_variance'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_6/moving_mean'variables/48/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_6/moving_variance'variables/49/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_7/moving_mean'variables/50/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_7/moving_variance'variables/51/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cdnn/batch_normalization_8/moving_mean'variables/52/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*cdnn/batch_normalization_8/moving_variance'variables/53/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcdnn/dense_9/kernel'variables/54/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEcdnn/dense_9/bias'variables/55/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
�
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17*
�
a0
b1
c2
d3
e4
f5
g6
h7
i8
j9
k10
l11
m12
n13
o14
p15
q16
r17
18
19
20*

�0
�1*
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
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
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

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
 bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

!kernel
"bias*
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
5moving_mean
6moving_variance*
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
7moving_mean
8moving_variance*
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
9moving_mean
:moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	)gamma
*beta
;moving_mean
<moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	+gamma
,beta
=moving_mean
>moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	-gamma
.beta
?moving_mean
@moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	/gamma
0beta
Amoving_mean
Bmoving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	1gamma
2beta
Cmoving_mean
Dmoving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	3gamma
4beta
Emoving_mean
Fmoving_variance*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

G0
H1*

G0
H1*
	
R0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
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
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37*
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
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

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
	
I0* 
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
0
1*

0
1*
	
J0* 
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
	
K0* 
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
	
L0* 
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
	
M0* 
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
	
N0* 
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
0
1*

0
1*
	
O0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
 1*

0
 1*
	
P0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

!0
"1*

!0
"1*
	
Q0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
 
#0
$1
52
63*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
%0
&1
72
83*

%0
&1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
'0
(1
92
:3*

'0
(1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
)0
*1
;2
<3*

)0
*1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
+0
,1
=2
>3*

+0
,1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
-0
.1
?2
@3*

-0
.1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
/0
01
A2
B3*

/0
01*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
10
21
C2
D3*

10
21*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
30
41
E2
F3*

30
41*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
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
	
R0* 
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
e_
VARIABLE_VALUESGD/m/cdnn/dense_6/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUESGD/m/cdnn/dense_6/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUESGD/m/cdnn/dense_7/kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUESGD/m/cdnn/dense_7/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUESGD/m/cdnn/dense_8/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUESGD/m/cdnn/dense_8/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$SGD/m/cdnn/batch_normalization/gamma2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#SGD/m/cdnn/batch_normalization/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_1/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_1/beta2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_2/gamma2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_2/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_3/gamma2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_3/beta2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_4/gamma2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_4/beta2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_5/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_5/beta2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_6/gamma2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_6/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_7/gamma2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_7/beta2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&SGD/m/cdnn/batch_normalization_8/gamma2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%SGD/m/cdnn/batch_normalization_8/beta2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUESGD/m/cdnn/dense_9/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUESGD/m/cdnn/dense_9/bias2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
	
I0* 
* 
* 
* 
* 
* 
* 
	
J0* 
* 
* 
* 
* 
* 
* 
	
K0* 
* 
* 
* 
* 
* 
* 
	
L0* 
* 
* 
* 
* 
* 
* 
	
M0* 
* 
* 
* 
* 
* 
* 
	
N0* 
* 
* 
* 
* 
* 
* 
	
O0* 
* 
* 
* 
* 
* 
* 
	
P0* 
* 
* 
* 
* 
* 
* 
	
Q0* 
* 
* 
* 

50
61*
* 
* 
* 
* 
* 
* 
* 
* 

70
81*
* 
* 
* 
* 
* 
* 
* 
* 

90
:1*
* 
* 
* 
* 
* 
* 
* 
* 

;0
<1*
* 
* 
* 
* 
* 
* 
* 
* 

=0
>1*
* 
* 
* 
* 
* 
* 
* 
* 

?0
@1*
* 
* 
* 
* 
* 
* 
* 
* 

A0
B1*
* 
* 
* 
* 
* 
* 
* 
* 

C0
D1*
* 
* 
* 
* 
* 
* 
* 
* 

E0
F1*
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
�*
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%cdnn/dense/kernel/Read/ReadVariableOp#cdnn/dense/bias/Read/ReadVariableOp'cdnn/dense_1/kernel/Read/ReadVariableOp%cdnn/dense_1/bias/Read/ReadVariableOp'cdnn/dense_2/kernel/Read/ReadVariableOp%cdnn/dense_2/bias/Read/ReadVariableOp'cdnn/dense_3/kernel/Read/ReadVariableOp%cdnn/dense_3/bias/Read/ReadVariableOp'cdnn/dense_4/kernel/Read/ReadVariableOp%cdnn/dense_4/bias/Read/ReadVariableOp'cdnn/dense_5/kernel/Read/ReadVariableOp%cdnn/dense_5/bias/Read/ReadVariableOp'cdnn/dense_6/kernel/Read/ReadVariableOp%cdnn/dense_6/bias/Read/ReadVariableOp'cdnn/dense_7/kernel/Read/ReadVariableOp%cdnn/dense_7/bias/Read/ReadVariableOp'cdnn/dense_8/kernel/Read/ReadVariableOp%cdnn/dense_8/bias/Read/ReadVariableOp2cdnn/batch_normalization/gamma/Read/ReadVariableOp1cdnn/batch_normalization/beta/Read/ReadVariableOp4cdnn/batch_normalization_1/gamma/Read/ReadVariableOp3cdnn/batch_normalization_1/beta/Read/ReadVariableOp4cdnn/batch_normalization_2/gamma/Read/ReadVariableOp3cdnn/batch_normalization_2/beta/Read/ReadVariableOp4cdnn/batch_normalization_3/gamma/Read/ReadVariableOp3cdnn/batch_normalization_3/beta/Read/ReadVariableOp4cdnn/batch_normalization_4/gamma/Read/ReadVariableOp3cdnn/batch_normalization_4/beta/Read/ReadVariableOp4cdnn/batch_normalization_5/gamma/Read/ReadVariableOp3cdnn/batch_normalization_5/beta/Read/ReadVariableOp4cdnn/batch_normalization_6/gamma/Read/ReadVariableOp3cdnn/batch_normalization_6/beta/Read/ReadVariableOp4cdnn/batch_normalization_7/gamma/Read/ReadVariableOp3cdnn/batch_normalization_7/beta/Read/ReadVariableOp4cdnn/batch_normalization_8/gamma/Read/ReadVariableOp3cdnn/batch_normalization_8/beta/Read/ReadVariableOp8cdnn/batch_normalization/moving_mean/Read/ReadVariableOp<cdnn/batch_normalization/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_1/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_1/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_2/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_2/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_3/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_3/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_4/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_4/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_5/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_5/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_6/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_6/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_7/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_7/moving_variance/Read/ReadVariableOp:cdnn/batch_normalization_8/moving_mean/Read/ReadVariableOp>cdnn/batch_normalization_8/moving_variance/Read/ReadVariableOp'cdnn/dense_9/kernel/Read/ReadVariableOp%cdnn/dense_9/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp+SGD/m/cdnn/dense/kernel/Read/ReadVariableOp)SGD/m/cdnn/dense/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_1/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_1/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_2/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_2/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_3/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_3/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_4/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_4/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_5/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_5/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_6/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_6/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_7/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_7/bias/Read/ReadVariableOp-SGD/m/cdnn/dense_8/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_8/bias/Read/ReadVariableOp8SGD/m/cdnn/batch_normalization/gamma/Read/ReadVariableOp7SGD/m/cdnn/batch_normalization/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_1/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_1/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_2/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_2/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_3/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_3/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_4/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_4/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_5/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_5/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_6/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_6/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_7/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_7/beta/Read/ReadVariableOp:SGD/m/cdnn/batch_normalization_8/gamma/Read/ReadVariableOp9SGD/m/cdnn/batch_normalization_8/beta/Read/ReadVariableOp-SGD/m/cdnn/dense_9/kernel/Read/ReadVariableOp+SGD/m/cdnn/dense_9/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*q
Tinj
h2f	*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_1092437
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecdnn/dense/kernelcdnn/dense/biascdnn/dense_1/kernelcdnn/dense_1/biascdnn/dense_2/kernelcdnn/dense_2/biascdnn/dense_3/kernelcdnn/dense_3/biascdnn/dense_4/kernelcdnn/dense_4/biascdnn/dense_5/kernelcdnn/dense_5/biascdnn/dense_6/kernelcdnn/dense_6/biascdnn/dense_7/kernelcdnn/dense_7/biascdnn/dense_8/kernelcdnn/dense_8/biascdnn/batch_normalization/gammacdnn/batch_normalization/beta cdnn/batch_normalization_1/gammacdnn/batch_normalization_1/beta cdnn/batch_normalization_2/gammacdnn/batch_normalization_2/beta cdnn/batch_normalization_3/gammacdnn/batch_normalization_3/beta cdnn/batch_normalization_4/gammacdnn/batch_normalization_4/beta cdnn/batch_normalization_5/gammacdnn/batch_normalization_5/beta cdnn/batch_normalization_6/gammacdnn/batch_normalization_6/beta cdnn/batch_normalization_7/gammacdnn/batch_normalization_7/beta cdnn/batch_normalization_8/gammacdnn/batch_normalization_8/beta$cdnn/batch_normalization/moving_mean(cdnn/batch_normalization/moving_variance&cdnn/batch_normalization_1/moving_mean*cdnn/batch_normalization_1/moving_variance&cdnn/batch_normalization_2/moving_mean*cdnn/batch_normalization_2/moving_variance&cdnn/batch_normalization_3/moving_mean*cdnn/batch_normalization_3/moving_variance&cdnn/batch_normalization_4/moving_mean*cdnn/batch_normalization_4/moving_variance&cdnn/batch_normalization_5/moving_mean*cdnn/batch_normalization_5/moving_variance&cdnn/batch_normalization_6/moving_mean*cdnn/batch_normalization_6/moving_variance&cdnn/batch_normalization_7/moving_mean*cdnn/batch_normalization_7/moving_variance&cdnn/batch_normalization_8/moving_mean*cdnn/batch_normalization_8/moving_variancecdnn/dense_9/kernelcdnn/dense_9/bias	iterationlearning_rateSGD/m/cdnn/dense/kernelSGD/m/cdnn/dense/biasSGD/m/cdnn/dense_1/kernelSGD/m/cdnn/dense_1/biasSGD/m/cdnn/dense_2/kernelSGD/m/cdnn/dense_2/biasSGD/m/cdnn/dense_3/kernelSGD/m/cdnn/dense_3/biasSGD/m/cdnn/dense_4/kernelSGD/m/cdnn/dense_4/biasSGD/m/cdnn/dense_5/kernelSGD/m/cdnn/dense_5/biasSGD/m/cdnn/dense_6/kernelSGD/m/cdnn/dense_6/biasSGD/m/cdnn/dense_7/kernelSGD/m/cdnn/dense_7/biasSGD/m/cdnn/dense_8/kernelSGD/m/cdnn/dense_8/bias$SGD/m/cdnn/batch_normalization/gamma#SGD/m/cdnn/batch_normalization/beta&SGD/m/cdnn/batch_normalization_1/gamma%SGD/m/cdnn/batch_normalization_1/beta&SGD/m/cdnn/batch_normalization_2/gamma%SGD/m/cdnn/batch_normalization_2/beta&SGD/m/cdnn/batch_normalization_3/gamma%SGD/m/cdnn/batch_normalization_3/beta&SGD/m/cdnn/batch_normalization_4/gamma%SGD/m/cdnn/batch_normalization_4/beta&SGD/m/cdnn/batch_normalization_5/gamma%SGD/m/cdnn/batch_normalization_5/beta&SGD/m/cdnn/batch_normalization_6/gamma%SGD/m/cdnn/batch_normalization_6/beta&SGD/m/cdnn/batch_normalization_7/gamma%SGD/m/cdnn/batch_normalization_7/beta&SGD/m/cdnn/batch_normalization_8/gamma%SGD/m/cdnn/batch_normalization_8/betaSGD/m/cdnn/dense_9/kernelSGD/m/cdnn/dense_9/biastotal_1count_1totalcount*p
Tini
g2e*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_1092747��"
�
�
7__inference_batch_normalization_4_layer_call_fn_1091727

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1088078o
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
D__inference_dense_5_layer_call_and_return_conditional_losses_1088636

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
�
�
'__inference_dense_layer_call_fn_1091187

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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1088486o
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
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1087996

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
__inference_loss_fn_8_1091073P
>cdnn_dense_8_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_8_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_8/kernel/Regularizer/L2LossL2Loss=cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_8/kernel/Regularizer/mulMul.cdnn/dense_8/kernel/Regularizer/mul/x:output:0/cdnn/dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'cdnn/dense_8/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp
�$
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1091474

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
D__inference_dense_1_layer_call_and_return_conditional_losses_1088516

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1091634

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
D__inference_dense_9_layer_call_and_return_conditional_losses_1088762

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_9/kernel/Regularizer/L2LossL2Loss=cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_9/kernel/Regularizer/mulMul.cdnn/dense_9/kernel/Regularizer/mul/x:output:0/cdnn/dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_dense_6_layer_call_and_return_conditional_losses_1088666

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
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
�
�
D__inference_dense_8_layer_call_and_return_conditional_losses_1088726

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_8/kernel/Regularizer/L2LossL2Loss=cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_8/kernel/Regularizer/mulMul.cdnn/dense_8/kernel/Regularizer/mul/x:output:0/cdnn/dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_2_1091019P
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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1087750

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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1091760

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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1088160

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
__inference_loss_fn_3_1091028P
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
�$
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1091554

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
7__inference_batch_normalization_1_layer_call_fn_1091500

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1087879o
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1088043

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
�
�
)__inference_dense_3_layer_call_fn_1091259

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
GPU 2J 8� *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1088576o
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1087879

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
�
�
)__inference_dense_5_layer_call_fn_1091307

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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1088636o
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
B__inference_dense_layer_call_and_return_conditional_losses_1088486

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
�
�
D__inference_dense_7_layer_call_and_return_conditional_losses_1088696

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_7/kernel/Regularizer/L2LossL2Loss=cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_7/kernel/Regularizer/mulMul.cdnn/dense_7/kernel/Regularizer/mul/x:output:0/cdnn/dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1091874

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
7__inference_batch_normalization_3_layer_call_fn_1091647

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1087996o
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
�
�
&__inference_cdnn_layer_call_fn_1090220
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

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:
identity��StatefulPartitionedCall�
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
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_cdnn_layer_call_and_return_conditional_losses_1088816o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1087914

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1087961

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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1091440

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
__inference_loss_fn_1_1091010P
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
��
�2
"__inference__wrapped_model_1087726
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
+cdnn_dense_6_matmul_readvariableop_resource::
,cdnn_dense_6_biasadd_readvariableop_resource:E
7cdnn_batch_normalization_6_cast_readvariableop_resource:G
9cdnn_batch_normalization_6_cast_1_readvariableop_resource:G
9cdnn_batch_normalization_6_cast_2_readvariableop_resource:G
9cdnn_batch_normalization_6_cast_3_readvariableop_resource:=
+cdnn_dense_7_matmul_readvariableop_resource::
,cdnn_dense_7_biasadd_readvariableop_resource:E
7cdnn_batch_normalization_7_cast_readvariableop_resource:G
9cdnn_batch_normalization_7_cast_1_readvariableop_resource:G
9cdnn_batch_normalization_7_cast_2_readvariableop_resource:G
9cdnn_batch_normalization_7_cast_3_readvariableop_resource:=
+cdnn_dense_8_matmul_readvariableop_resource::
,cdnn_dense_8_biasadd_readvariableop_resource:E
7cdnn_batch_normalization_8_cast_readvariableop_resource:G
9cdnn_batch_normalization_8_cast_1_readvariableop_resource:G
9cdnn_batch_normalization_8_cast_2_readvariableop_resource:G
9cdnn_batch_normalization_8_cast_3_readvariableop_resource:=
+cdnn_dense_9_matmul_readvariableop_resource::
,cdnn_dense_9_biasadd_readvariableop_resource:
identity��,cdnn/batch_normalization/Cast/ReadVariableOp�.cdnn/batch_normalization/Cast_1/ReadVariableOp�.cdnn/batch_normalization/Cast_2/ReadVariableOp�.cdnn/batch_normalization/Cast_3/ReadVariableOp�.cdnn/batch_normalization_1/Cast/ReadVariableOp�0cdnn/batch_normalization_1/Cast_1/ReadVariableOp�0cdnn/batch_normalization_1/Cast_2/ReadVariableOp�0cdnn/batch_normalization_1/Cast_3/ReadVariableOp�.cdnn/batch_normalization_2/Cast/ReadVariableOp�0cdnn/batch_normalization_2/Cast_1/ReadVariableOp�0cdnn/batch_normalization_2/Cast_2/ReadVariableOp�0cdnn/batch_normalization_2/Cast_3/ReadVariableOp�.cdnn/batch_normalization_3/Cast/ReadVariableOp�0cdnn/batch_normalization_3/Cast_1/ReadVariableOp�0cdnn/batch_normalization_3/Cast_2/ReadVariableOp�0cdnn/batch_normalization_3/Cast_3/ReadVariableOp�.cdnn/batch_normalization_4/Cast/ReadVariableOp�0cdnn/batch_normalization_4/Cast_1/ReadVariableOp�0cdnn/batch_normalization_4/Cast_2/ReadVariableOp�0cdnn/batch_normalization_4/Cast_3/ReadVariableOp�.cdnn/batch_normalization_5/Cast/ReadVariableOp�0cdnn/batch_normalization_5/Cast_1/ReadVariableOp�0cdnn/batch_normalization_5/Cast_2/ReadVariableOp�0cdnn/batch_normalization_5/Cast_3/ReadVariableOp�.cdnn/batch_normalization_6/Cast/ReadVariableOp�0cdnn/batch_normalization_6/Cast_1/ReadVariableOp�0cdnn/batch_normalization_6/Cast_2/ReadVariableOp�0cdnn/batch_normalization_6/Cast_3/ReadVariableOp�.cdnn/batch_normalization_7/Cast/ReadVariableOp�0cdnn/batch_normalization_7/Cast_1/ReadVariableOp�0cdnn/batch_normalization_7/Cast_2/ReadVariableOp�0cdnn/batch_normalization_7/Cast_3/ReadVariableOp�.cdnn/batch_normalization_8/Cast/ReadVariableOp�0cdnn/batch_normalization_8/Cast_1/ReadVariableOp�0cdnn/batch_normalization_8/Cast_2/ReadVariableOp�0cdnn/batch_normalization_8/Cast_3/ReadVariableOp�!cdnn/dense/BiasAdd/ReadVariableOp� cdnn/dense/MatMul/ReadVariableOp�#cdnn/dense_1/BiasAdd/ReadVariableOp�"cdnn/dense_1/MatMul/ReadVariableOp�#cdnn/dense_2/BiasAdd/ReadVariableOp�"cdnn/dense_2/MatMul/ReadVariableOp�#cdnn/dense_3/BiasAdd/ReadVariableOp�"cdnn/dense_3/MatMul/ReadVariableOp�#cdnn/dense_4/BiasAdd/ReadVariableOp�"cdnn/dense_4/MatMul/ReadVariableOp�#cdnn/dense_5/BiasAdd/ReadVariableOp�"cdnn/dense_5/MatMul/ReadVariableOp�#cdnn/dense_6/BiasAdd/ReadVariableOp�"cdnn/dense_6/MatMul/ReadVariableOp�#cdnn/dense_7/BiasAdd/ReadVariableOp�"cdnn/dense_7/MatMul/ReadVariableOp�#cdnn/dense_8/BiasAdd/ReadVariableOp�"cdnn/dense_8/MatMul/ReadVariableOp�#cdnn/dense_9/BiasAdd/ReadVariableOp�"cdnn/dense_9/MatMul/ReadVariableOp�
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
"cdnn/dense_6/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_6/MatMulMatMul.cdnn/batch_normalization_5/batchnorm/add_1:z:0*cdnn/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_6/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_6/BiasAddBiasAddcdnn/dense_6/MatMul:product:0+cdnn/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
cdnn/dense_6/ReluRelucdnn/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.cdnn/batch_normalization_6/Cast/ReadVariableOpReadVariableOp7cdnn_batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp9cdnn_batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp9cdnn_batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp9cdnn_batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0o
*cdnn/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(cdnn/batch_normalization_6/batchnorm/addAddV28cdnn/batch_normalization_6/Cast_1/ReadVariableOp:value:03cdnn/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_6/batchnorm/RsqrtRsqrt,cdnn/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_6/batchnorm/mulMul.cdnn/batch_normalization_6/batchnorm/Rsqrt:y:08cdnn/batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_6/batchnorm/mul_1Mulcdnn/dense_6/Relu:activations:0,cdnn/batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
*cdnn/batch_normalization_6/batchnorm/mul_2Mul6cdnn/batch_normalization_6/Cast/ReadVariableOp:value:0,cdnn/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_6/batchnorm/subSub8cdnn/batch_normalization_6/Cast_2/ReadVariableOp:value:0.cdnn/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_6/batchnorm/add_1AddV2.cdnn/batch_normalization_6/batchnorm/mul_1:z:0,cdnn/batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
"cdnn/dense_7/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_7/MatMulMatMul.cdnn/batch_normalization_6/batchnorm/add_1:z:0*cdnn/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_7/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_7/BiasAddBiasAddcdnn/dense_7/MatMul:product:0+cdnn/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
cdnn/dense_7/ReluRelucdnn/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.cdnn/batch_normalization_7/Cast/ReadVariableOpReadVariableOp7cdnn_batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp9cdnn_batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp9cdnn_batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp9cdnn_batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0o
*cdnn/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(cdnn/batch_normalization_7/batchnorm/addAddV28cdnn/batch_normalization_7/Cast_1/ReadVariableOp:value:03cdnn/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_7/batchnorm/RsqrtRsqrt,cdnn/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_7/batchnorm/mulMul.cdnn/batch_normalization_7/batchnorm/Rsqrt:y:08cdnn/batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_7/batchnorm/mul_1Mulcdnn/dense_7/Relu:activations:0,cdnn/batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
*cdnn/batch_normalization_7/batchnorm/mul_2Mul6cdnn/batch_normalization_7/Cast/ReadVariableOp:value:0,cdnn/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_7/batchnorm/subSub8cdnn/batch_normalization_7/Cast_2/ReadVariableOp:value:0.cdnn/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_7/batchnorm/add_1AddV2.cdnn/batch_normalization_7/batchnorm/mul_1:z:0,cdnn/batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
"cdnn/dense_8/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_8/MatMulMatMul.cdnn/batch_normalization_7/batchnorm/add_1:z:0*cdnn/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_8/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_8/BiasAddBiasAddcdnn/dense_8/MatMul:product:0+cdnn/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
cdnn/dense_8/ReluRelucdnn/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.cdnn/batch_normalization_8/Cast/ReadVariableOpReadVariableOp7cdnn_batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp9cdnn_batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp9cdnn_batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
0cdnn/batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp9cdnn_batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0o
*cdnn/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(cdnn/batch_normalization_8/batchnorm/addAddV28cdnn/batch_normalization_8/Cast_1/ReadVariableOp:value:03cdnn/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_8/batchnorm/RsqrtRsqrt,cdnn/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_8/batchnorm/mulMul.cdnn/batch_normalization_8/batchnorm/Rsqrt:y:08cdnn/batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_8/batchnorm/mul_1Mulcdnn/dense_8/Relu:activations:0,cdnn/batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
*cdnn/batch_normalization_8/batchnorm/mul_2Mul6cdnn/batch_normalization_8/Cast/ReadVariableOp:value:0,cdnn/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:�
(cdnn/batch_normalization_8/batchnorm/subSub8cdnn/batch_normalization_8/Cast_2/ReadVariableOp:value:0.cdnn/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
*cdnn/batch_normalization_8/batchnorm/add_1AddV2.cdnn/batch_normalization_8/batchnorm/mul_1:z:0,cdnn/batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
cdnn/dropout/IdentityIdentity.cdnn/batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:����������
"cdnn/dense_9/MatMul/ReadVariableOpReadVariableOp+cdnn_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
cdnn/dense_9/MatMulMatMulcdnn/dropout/Identity:output:0*cdnn/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#cdnn/dense_9/BiasAdd/ReadVariableOpReadVariableOp,cdnn_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cdnn/dense_9/BiasAddBiasAddcdnn/dense_9/MatMul:product:0+cdnn/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s
cdnn/activation/SigmoidSigmoidcdnn/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentitycdnn/activation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^cdnn/batch_normalization/Cast/ReadVariableOp/^cdnn/batch_normalization/Cast_1/ReadVariableOp/^cdnn/batch_normalization/Cast_2/ReadVariableOp/^cdnn/batch_normalization/Cast_3/ReadVariableOp/^cdnn/batch_normalization_1/Cast/ReadVariableOp1^cdnn/batch_normalization_1/Cast_1/ReadVariableOp1^cdnn/batch_normalization_1/Cast_2/ReadVariableOp1^cdnn/batch_normalization_1/Cast_3/ReadVariableOp/^cdnn/batch_normalization_2/Cast/ReadVariableOp1^cdnn/batch_normalization_2/Cast_1/ReadVariableOp1^cdnn/batch_normalization_2/Cast_2/ReadVariableOp1^cdnn/batch_normalization_2/Cast_3/ReadVariableOp/^cdnn/batch_normalization_3/Cast/ReadVariableOp1^cdnn/batch_normalization_3/Cast_1/ReadVariableOp1^cdnn/batch_normalization_3/Cast_2/ReadVariableOp1^cdnn/batch_normalization_3/Cast_3/ReadVariableOp/^cdnn/batch_normalization_4/Cast/ReadVariableOp1^cdnn/batch_normalization_4/Cast_1/ReadVariableOp1^cdnn/batch_normalization_4/Cast_2/ReadVariableOp1^cdnn/batch_normalization_4/Cast_3/ReadVariableOp/^cdnn/batch_normalization_5/Cast/ReadVariableOp1^cdnn/batch_normalization_5/Cast_1/ReadVariableOp1^cdnn/batch_normalization_5/Cast_2/ReadVariableOp1^cdnn/batch_normalization_5/Cast_3/ReadVariableOp/^cdnn/batch_normalization_6/Cast/ReadVariableOp1^cdnn/batch_normalization_6/Cast_1/ReadVariableOp1^cdnn/batch_normalization_6/Cast_2/ReadVariableOp1^cdnn/batch_normalization_6/Cast_3/ReadVariableOp/^cdnn/batch_normalization_7/Cast/ReadVariableOp1^cdnn/batch_normalization_7/Cast_1/ReadVariableOp1^cdnn/batch_normalization_7/Cast_2/ReadVariableOp1^cdnn/batch_normalization_7/Cast_3/ReadVariableOp/^cdnn/batch_normalization_8/Cast/ReadVariableOp1^cdnn/batch_normalization_8/Cast_1/ReadVariableOp1^cdnn/batch_normalization_8/Cast_2/ReadVariableOp1^cdnn/batch_normalization_8/Cast_3/ReadVariableOp"^cdnn/dense/BiasAdd/ReadVariableOp!^cdnn/dense/MatMul/ReadVariableOp$^cdnn/dense_1/BiasAdd/ReadVariableOp#^cdnn/dense_1/MatMul/ReadVariableOp$^cdnn/dense_2/BiasAdd/ReadVariableOp#^cdnn/dense_2/MatMul/ReadVariableOp$^cdnn/dense_3/BiasAdd/ReadVariableOp#^cdnn/dense_3/MatMul/ReadVariableOp$^cdnn/dense_4/BiasAdd/ReadVariableOp#^cdnn/dense_4/MatMul/ReadVariableOp$^cdnn/dense_5/BiasAdd/ReadVariableOp#^cdnn/dense_5/MatMul/ReadVariableOp$^cdnn/dense_6/BiasAdd/ReadVariableOp#^cdnn/dense_6/MatMul/ReadVariableOp$^cdnn/dense_7/BiasAdd/ReadVariableOp#^cdnn/dense_7/MatMul/ReadVariableOp$^cdnn/dense_8/BiasAdd/ReadVariableOp#^cdnn/dense_8/MatMul/ReadVariableOp$^cdnn/dense_9/BiasAdd/ReadVariableOp#^cdnn/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
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
0cdnn/batch_normalization_5/Cast_3/ReadVariableOp0cdnn/batch_normalization_5/Cast_3/ReadVariableOp2`
.cdnn/batch_normalization_6/Cast/ReadVariableOp.cdnn/batch_normalization_6/Cast/ReadVariableOp2d
0cdnn/batch_normalization_6/Cast_1/ReadVariableOp0cdnn/batch_normalization_6/Cast_1/ReadVariableOp2d
0cdnn/batch_normalization_6/Cast_2/ReadVariableOp0cdnn/batch_normalization_6/Cast_2/ReadVariableOp2d
0cdnn/batch_normalization_6/Cast_3/ReadVariableOp0cdnn/batch_normalization_6/Cast_3/ReadVariableOp2`
.cdnn/batch_normalization_7/Cast/ReadVariableOp.cdnn/batch_normalization_7/Cast/ReadVariableOp2d
0cdnn/batch_normalization_7/Cast_1/ReadVariableOp0cdnn/batch_normalization_7/Cast_1/ReadVariableOp2d
0cdnn/batch_normalization_7/Cast_2/ReadVariableOp0cdnn/batch_normalization_7/Cast_2/ReadVariableOp2d
0cdnn/batch_normalization_7/Cast_3/ReadVariableOp0cdnn/batch_normalization_7/Cast_3/ReadVariableOp2`
.cdnn/batch_normalization_8/Cast/ReadVariableOp.cdnn/batch_normalization_8/Cast/ReadVariableOp2d
0cdnn/batch_normalization_8/Cast_1/ReadVariableOp0cdnn/batch_normalization_8/Cast_1/ReadVariableOp2d
0cdnn/batch_normalization_8/Cast_2/ReadVariableOp0cdnn/batch_normalization_8/Cast_2/ReadVariableOp2d
0cdnn/batch_normalization_8/Cast_3/ReadVariableOp0cdnn/batch_normalization_8/Cast_3/ReadVariableOp2F
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
"cdnn/dense_6/MatMul/ReadVariableOp"cdnn/dense_6/MatMul/ReadVariableOp2J
#cdnn/dense_7/BiasAdd/ReadVariableOp#cdnn/dense_7/BiasAdd/ReadVariableOp2H
"cdnn/dense_7/MatMul/ReadVariableOp"cdnn/dense_7/MatMul/ReadVariableOp2J
#cdnn/dense_8/BiasAdd/ReadVariableOp#cdnn/dense_8/BiasAdd/ReadVariableOp2H
"cdnn/dense_8/MatMul/ReadVariableOp"cdnn/dense_8/MatMul/ReadVariableOp2J
#cdnn/dense_9/BiasAdd/ReadVariableOp#cdnn/dense_9/BiasAdd/ReadVariableOp2H
"cdnn/dense_9/MatMul/ReadVariableOp"cdnn/dense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
D__inference_dense_4_layer_call_and_return_conditional_losses_1088606

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
�
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1088324

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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1092000

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

c
D__inference_dropout_layer_call_and_return_conditional_losses_1091145

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?d
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
 *���>�
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1091840

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

c
D__inference_dropout_layer_call_and_return_conditional_losses_1088967

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?d
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
 *���>�
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
Ҵ
�0
 __inference__traced_save_1092437
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
,savev2_cdnn_dense_5_bias_read_readvariableop2
.savev2_cdnn_dense_6_kernel_read_readvariableop0
,savev2_cdnn_dense_6_bias_read_readvariableop2
.savev2_cdnn_dense_7_kernel_read_readvariableop0
,savev2_cdnn_dense_7_bias_read_readvariableop2
.savev2_cdnn_dense_8_kernel_read_readvariableop0
,savev2_cdnn_dense_8_bias_read_readvariableop=
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
:savev2_cdnn_batch_normalization_5_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_6_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_6_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_7_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_7_beta_read_readvariableop?
;savev2_cdnn_batch_normalization_8_gamma_read_readvariableop>
:savev2_cdnn_batch_normalization_8_beta_read_readvariableopC
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
Esavev2_cdnn_batch_normalization_5_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_6_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_6_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_7_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_7_moving_variance_read_readvariableopE
Asavev2_cdnn_batch_normalization_8_moving_mean_read_readvariableopI
Esavev2_cdnn_batch_normalization_8_moving_variance_read_readvariableop2
.savev2_cdnn_dense_9_kernel_read_readvariableop0
,savev2_cdnn_dense_9_bias_read_readvariableop(
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
2savev2_sgd_m_cdnn_dense_5_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_6_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_6_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_7_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_7_bias_read_readvariableop8
4savev2_sgd_m_cdnn_dense_8_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_8_bias_read_readvariableopC
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
@savev2_sgd_m_cdnn_batch_normalization_5_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_6_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_6_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_7_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_7_beta_read_readvariableopE
Asavev2_sgd_m_cdnn_batch_normalization_8_gamma_read_readvariableopD
@savev2_sgd_m_cdnn_batch_normalization_8_beta_read_readvariableop8
4savev2_sgd_m_cdnn_dense_9_kernel_read_readvariableop6
2savev2_sgd_m_cdnn_dense_9_bias_read_readvariableop&
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
: �$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*�$
value�$B�$eB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*�
value�B�eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �/
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_cdnn_dense_kernel_read_readvariableop*savev2_cdnn_dense_bias_read_readvariableop.savev2_cdnn_dense_1_kernel_read_readvariableop,savev2_cdnn_dense_1_bias_read_readvariableop.savev2_cdnn_dense_2_kernel_read_readvariableop,savev2_cdnn_dense_2_bias_read_readvariableop.savev2_cdnn_dense_3_kernel_read_readvariableop,savev2_cdnn_dense_3_bias_read_readvariableop.savev2_cdnn_dense_4_kernel_read_readvariableop,savev2_cdnn_dense_4_bias_read_readvariableop.savev2_cdnn_dense_5_kernel_read_readvariableop,savev2_cdnn_dense_5_bias_read_readvariableop.savev2_cdnn_dense_6_kernel_read_readvariableop,savev2_cdnn_dense_6_bias_read_readvariableop.savev2_cdnn_dense_7_kernel_read_readvariableop,savev2_cdnn_dense_7_bias_read_readvariableop.savev2_cdnn_dense_8_kernel_read_readvariableop,savev2_cdnn_dense_8_bias_read_readvariableop9savev2_cdnn_batch_normalization_gamma_read_readvariableop8savev2_cdnn_batch_normalization_beta_read_readvariableop;savev2_cdnn_batch_normalization_1_gamma_read_readvariableop:savev2_cdnn_batch_normalization_1_beta_read_readvariableop;savev2_cdnn_batch_normalization_2_gamma_read_readvariableop:savev2_cdnn_batch_normalization_2_beta_read_readvariableop;savev2_cdnn_batch_normalization_3_gamma_read_readvariableop:savev2_cdnn_batch_normalization_3_beta_read_readvariableop;savev2_cdnn_batch_normalization_4_gamma_read_readvariableop:savev2_cdnn_batch_normalization_4_beta_read_readvariableop;savev2_cdnn_batch_normalization_5_gamma_read_readvariableop:savev2_cdnn_batch_normalization_5_beta_read_readvariableop;savev2_cdnn_batch_normalization_6_gamma_read_readvariableop:savev2_cdnn_batch_normalization_6_beta_read_readvariableop;savev2_cdnn_batch_normalization_7_gamma_read_readvariableop:savev2_cdnn_batch_normalization_7_beta_read_readvariableop;savev2_cdnn_batch_normalization_8_gamma_read_readvariableop:savev2_cdnn_batch_normalization_8_beta_read_readvariableop?savev2_cdnn_batch_normalization_moving_mean_read_readvariableopCsavev2_cdnn_batch_normalization_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_1_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_1_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_2_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_2_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_3_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_3_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_4_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_4_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_5_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_5_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_6_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_6_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_7_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_7_moving_variance_read_readvariableopAsavev2_cdnn_batch_normalization_8_moving_mean_read_readvariableopEsavev2_cdnn_batch_normalization_8_moving_variance_read_readvariableop.savev2_cdnn_dense_9_kernel_read_readvariableop,savev2_cdnn_dense_9_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop2savev2_sgd_m_cdnn_dense_kernel_read_readvariableop0savev2_sgd_m_cdnn_dense_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_1_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_1_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_2_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_2_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_3_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_3_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_4_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_4_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_5_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_5_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_6_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_6_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_7_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_7_bias_read_readvariableop4savev2_sgd_m_cdnn_dense_8_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_8_bias_read_readvariableop?savev2_sgd_m_cdnn_batch_normalization_gamma_read_readvariableop>savev2_sgd_m_cdnn_batch_normalization_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_1_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_1_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_2_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_2_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_3_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_3_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_4_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_4_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_5_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_5_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_6_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_6_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_7_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_7_beta_read_readvariableopAsavev2_sgd_m_cdnn_batch_normalization_8_gamma_read_readvariableop@savev2_sgd_m_cdnn_batch_normalization_8_beta_read_readvariableop4savev2_sgd_m_cdnn_dense_9_kernel_read_readvariableop2savev2_sgd_m_cdnn_dense_9_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *s
dtypesi
g2e	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::: : ::::::::::::::::::::::::::::::::::::::: : : : : 2(
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
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 
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
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
::$7 

_output_shapes

:: 8

_output_shapes
::9

_output_shapes
: ::

_output_shapes
: :$; 

_output_shapes

:: <

_output_shapes
::$= 

_output_shapes

:: >

_output_shapes
::$? 

_output_shapes

:: @

_output_shapes
::$A 

_output_shapes

:: B

_output_shapes
::$C 

_output_shapes

:: D

_output_shapes
::$E 

_output_shapes

:: F

_output_shapes
::$G 

_output_shapes

:: H

_output_shapes
::$I 

_output_shapes

:: J

_output_shapes
::$K 

_output_shapes

:: L

_output_shapes
:: M

_output_shapes
:: N

_output_shapes
:: O

_output_shapes
:: P

_output_shapes
:: Q

_output_shapes
:: R

_output_shapes
:: S

_output_shapes
:: T

_output_shapes
:: U

_output_shapes
:: V

_output_shapes
:: W

_output_shapes
:: X

_output_shapes
:: Y

_output_shapes
:: Z

_output_shapes
:: [

_output_shapes
:: \

_output_shapes
:: ]

_output_shapes
:: ^

_output_shapes
::$_ 

_output_shapes

:: `

_output_shapes
::a

_output_shapes
: :b

_output_shapes
: :c

_output_shapes
: :d

_output_shapes
: :e

_output_shapes
: 
�
�
7__inference_batch_normalization_3_layer_call_fn_1091660

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1088043o
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1088207

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
�
__inference_loss_fn_4_1091037P
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
�	
�
__inference_loss_fn_7_1091064P
>cdnn_dense_7_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_7_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_7/kernel/Regularizer/L2LossL2Loss=cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_7/kernel/Regularizer/mulMul.cdnn/dense_7/kernel/Regularizer/mul/x:output:0/cdnn/dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'cdnn/dense_7/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
5__inference_batch_normalization_layer_call_fn_1091407

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
GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1087750o
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
7__inference_batch_normalization_8_layer_call_fn_1092060

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1088453o
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
7__inference_batch_normalization_7_layer_call_fn_1091980

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1088371o
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
__inference_loss_fn_6_1091055P
>cdnn_dense_6_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_6_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
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
�
�
D__inference_dense_2_layer_call_and_return_conditional_losses_1091250

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
__inference_loss_fn_0_1091001N
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
�
�
7__inference_batch_normalization_7_layer_call_fn_1091967

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1088324o
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1091794

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
�$
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1088371

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
�
�
&__inference_cdnn_layer_call_fn_1089588
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

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:
identity��StatefulPartitionedCall�
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
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(& #$%&)*+,/0125678*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_cdnn_layer_call_and_return_conditional_losses_1089356o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
D__inference_dense_9_layer_call_and_return_conditional_losses_1091168

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_9/kernel/Regularizer/L2LossL2Loss=cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_9/kernel/Regularizer/mulMul.cdnn/dense_9/kernel/Regularizer/mul/x:output:0/cdnn/dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1091680

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
)__inference_dense_8_layer_call_fn_1091379

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
GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1088726o
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
��
�2
A__inference_cdnn_layer_call_and_return_conditional_losses_1090596
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
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:@
2batch_normalization_6_cast_readvariableop_resource:B
4batch_normalization_6_cast_1_readvariableop_resource:B
4batch_normalization_6_cast_2_readvariableop_resource:B
4batch_normalization_6_cast_3_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:B
4batch_normalization_7_cast_2_readvariableop_resource:B
4batch_normalization_7_cast_3_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:B
4batch_normalization_8_cast_2_readvariableop_resource:B
4batch_normalization_8_cast_3_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity��'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�)batch_normalization/Cast_2/ReadVariableOp�)batch_normalization/Cast_3/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�+batch_normalization_1/Cast_2/ReadVariableOp�+batch_normalization_1/Cast_3/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�+batch_normalization_2/Cast_2/ReadVariableOp�+batch_normalization_2/Cast_3/ReadVariableOp�)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�+batch_normalization_3/Cast_2/ReadVariableOp�+batch_normalization_3/Cast_3/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�+batch_normalization_4/Cast_2/ReadVariableOp�+batch_normalization_4/Cast_3/ReadVariableOp�)batch_normalization_5/Cast/ReadVariableOp�+batch_normalization_5/Cast_1/ReadVariableOp�+batch_normalization_5/Cast_2/ReadVariableOp�+batch_normalization_5/Cast_3/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�+batch_normalization_6/Cast_2/ReadVariableOp�+batch_normalization_6/Cast_3/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�+batch_normalization_7/Cast_2/ReadVariableOp�+batch_normalization_7/Cast_3/ReadVariableOp�)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�+batch_normalization_8/Cast_2/ReadVariableOp�+batch_normalization_8/Cast_3/ReadVariableOp�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
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
:����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:����������
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_6/batchnorm/mul_1Muldense_6/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_7/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:����������
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_7/batchnorm/mul_1Muldense_7/Relu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_8/MatMulMatMul)batch_normalization_7/batchnorm/add_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:����������
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_8/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������y
dropout/IdentityIdentity)batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:����������
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_9/MatMulMatMuldropout/Identity:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
activation/SigmoidSigmoiddense_9/BiasAdd:output:0*
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

:*
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
: �
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_7/kernel/Regularizer/L2LossL2Loss=cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_7/kernel/Regularizer/mulMul.cdnn/dense_7/kernel/Regularizer/mul/x:output:0/cdnn/dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_8/kernel/Regularizer/L2LossL2Loss=cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_8/kernel/Regularizer/mulMul.cdnn/dense_8/kernel/Regularizer/mul/x:output:0/cdnn/dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_9/kernel/Regularizer/L2LossL2Loss=cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_9/kernel/Regularizer/mulMul.cdnn/dense_9/kernel/Regularizer/mul/x:output:0/cdnn/dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp,^batch_normalization_3/Cast_2/ReadVariableOp,^batch_normalization_3/Cast_3/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp,^batch_normalization_5/Cast_2/ReadVariableOp,^batch_normalization_5/Cast_3/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
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
+batch_normalization_5/Cast_3/ReadVariableOp+batch_normalization_5/Cast_3/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2Z
+batch_normalization_6/Cast_2/ReadVariableOp+batch_normalization_6/Cast_2/ReadVariableOp2Z
+batch_normalization_6/Cast_3/ReadVariableOp+batch_normalization_6/Cast_3/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2Z
+batch_normalization_7/Cast_2/ReadVariableOp+batch_normalization_7/Cast_2/ReadVariableOp2Z
+batch_normalization_7/Cast_3/ReadVariableOp+batch_normalization_7/Cast_3/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2Z
+batch_normalization_8/Cast_2/ReadVariableOp+batch_normalization_8/Cast_2/ReadVariableOp2Z
+batch_normalization_8/Cast_3/ReadVariableOp+batch_normalization_8/Cast_3/ReadVariableOp2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp2<
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
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
��
�D
#__inference__traced_restore_1092747
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
%assignvariableop_11_cdnn_dense_5_bias:9
'assignvariableop_12_cdnn_dense_6_kernel:3
%assignvariableop_13_cdnn_dense_6_bias:9
'assignvariableop_14_cdnn_dense_7_kernel:3
%assignvariableop_15_cdnn_dense_7_bias:9
'assignvariableop_16_cdnn_dense_8_kernel:3
%assignvariableop_17_cdnn_dense_8_bias:@
2assignvariableop_18_cdnn_batch_normalization_gamma:?
1assignvariableop_19_cdnn_batch_normalization_beta:B
4assignvariableop_20_cdnn_batch_normalization_1_gamma:A
3assignvariableop_21_cdnn_batch_normalization_1_beta:B
4assignvariableop_22_cdnn_batch_normalization_2_gamma:A
3assignvariableop_23_cdnn_batch_normalization_2_beta:B
4assignvariableop_24_cdnn_batch_normalization_3_gamma:A
3assignvariableop_25_cdnn_batch_normalization_3_beta:B
4assignvariableop_26_cdnn_batch_normalization_4_gamma:A
3assignvariableop_27_cdnn_batch_normalization_4_beta:B
4assignvariableop_28_cdnn_batch_normalization_5_gamma:A
3assignvariableop_29_cdnn_batch_normalization_5_beta:B
4assignvariableop_30_cdnn_batch_normalization_6_gamma:A
3assignvariableop_31_cdnn_batch_normalization_6_beta:B
4assignvariableop_32_cdnn_batch_normalization_7_gamma:A
3assignvariableop_33_cdnn_batch_normalization_7_beta:B
4assignvariableop_34_cdnn_batch_normalization_8_gamma:A
3assignvariableop_35_cdnn_batch_normalization_8_beta:F
8assignvariableop_36_cdnn_batch_normalization_moving_mean:J
<assignvariableop_37_cdnn_batch_normalization_moving_variance:H
:assignvariableop_38_cdnn_batch_normalization_1_moving_mean:L
>assignvariableop_39_cdnn_batch_normalization_1_moving_variance:H
:assignvariableop_40_cdnn_batch_normalization_2_moving_mean:L
>assignvariableop_41_cdnn_batch_normalization_2_moving_variance:H
:assignvariableop_42_cdnn_batch_normalization_3_moving_mean:L
>assignvariableop_43_cdnn_batch_normalization_3_moving_variance:H
:assignvariableop_44_cdnn_batch_normalization_4_moving_mean:L
>assignvariableop_45_cdnn_batch_normalization_4_moving_variance:H
:assignvariableop_46_cdnn_batch_normalization_5_moving_mean:L
>assignvariableop_47_cdnn_batch_normalization_5_moving_variance:H
:assignvariableop_48_cdnn_batch_normalization_6_moving_mean:L
>assignvariableop_49_cdnn_batch_normalization_6_moving_variance:H
:assignvariableop_50_cdnn_batch_normalization_7_moving_mean:L
>assignvariableop_51_cdnn_batch_normalization_7_moving_variance:H
:assignvariableop_52_cdnn_batch_normalization_8_moving_mean:L
>assignvariableop_53_cdnn_batch_normalization_8_moving_variance:9
'assignvariableop_54_cdnn_dense_9_kernel:3
%assignvariableop_55_cdnn_dense_9_bias:'
assignvariableop_56_iteration:	 +
!assignvariableop_57_learning_rate: =
+assignvariableop_58_sgd_m_cdnn_dense_kernel:7
)assignvariableop_59_sgd_m_cdnn_dense_bias:?
-assignvariableop_60_sgd_m_cdnn_dense_1_kernel:9
+assignvariableop_61_sgd_m_cdnn_dense_1_bias:?
-assignvariableop_62_sgd_m_cdnn_dense_2_kernel:9
+assignvariableop_63_sgd_m_cdnn_dense_2_bias:?
-assignvariableop_64_sgd_m_cdnn_dense_3_kernel:9
+assignvariableop_65_sgd_m_cdnn_dense_3_bias:?
-assignvariableop_66_sgd_m_cdnn_dense_4_kernel:9
+assignvariableop_67_sgd_m_cdnn_dense_4_bias:?
-assignvariableop_68_sgd_m_cdnn_dense_5_kernel:9
+assignvariableop_69_sgd_m_cdnn_dense_5_bias:?
-assignvariableop_70_sgd_m_cdnn_dense_6_kernel:9
+assignvariableop_71_sgd_m_cdnn_dense_6_bias:?
-assignvariableop_72_sgd_m_cdnn_dense_7_kernel:9
+assignvariableop_73_sgd_m_cdnn_dense_7_bias:?
-assignvariableop_74_sgd_m_cdnn_dense_8_kernel:9
+assignvariableop_75_sgd_m_cdnn_dense_8_bias:F
8assignvariableop_76_sgd_m_cdnn_batch_normalization_gamma:E
7assignvariableop_77_sgd_m_cdnn_batch_normalization_beta:H
:assignvariableop_78_sgd_m_cdnn_batch_normalization_1_gamma:G
9assignvariableop_79_sgd_m_cdnn_batch_normalization_1_beta:H
:assignvariableop_80_sgd_m_cdnn_batch_normalization_2_gamma:G
9assignvariableop_81_sgd_m_cdnn_batch_normalization_2_beta:H
:assignvariableop_82_sgd_m_cdnn_batch_normalization_3_gamma:G
9assignvariableop_83_sgd_m_cdnn_batch_normalization_3_beta:H
:assignvariableop_84_sgd_m_cdnn_batch_normalization_4_gamma:G
9assignvariableop_85_sgd_m_cdnn_batch_normalization_4_beta:H
:assignvariableop_86_sgd_m_cdnn_batch_normalization_5_gamma:G
9assignvariableop_87_sgd_m_cdnn_batch_normalization_5_beta:H
:assignvariableop_88_sgd_m_cdnn_batch_normalization_6_gamma:G
9assignvariableop_89_sgd_m_cdnn_batch_normalization_6_beta:H
:assignvariableop_90_sgd_m_cdnn_batch_normalization_7_gamma:G
9assignvariableop_91_sgd_m_cdnn_batch_normalization_7_beta:H
:assignvariableop_92_sgd_m_cdnn_batch_normalization_8_gamma:G
9assignvariableop_93_sgd_m_cdnn_batch_normalization_8_beta:?
-assignvariableop_94_sgd_m_cdnn_dense_9_kernel:9
+assignvariableop_95_sgd_m_cdnn_dense_9_bias:%
assignvariableop_96_total_1: %
assignvariableop_97_count_1: #
assignvariableop_98_total: #
assignvariableop_99_count: 
identity_101��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*�$
value�$B�$eB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*�
value�B�eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*s
dtypesi
g2e	[
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
AssignVariableOp_12AssignVariableOp'assignvariableop_12_cdnn_dense_6_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp%assignvariableop_13_cdnn_dense_6_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp'assignvariableop_14_cdnn_dense_7_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp%assignvariableop_15_cdnn_dense_7_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_cdnn_dense_8_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp%assignvariableop_17_cdnn_dense_8_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp2assignvariableop_18_cdnn_batch_normalization_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp1assignvariableop_19_cdnn_batch_normalization_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp4assignvariableop_20_cdnn_batch_normalization_1_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp3assignvariableop_21_cdnn_batch_normalization_1_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_cdnn_batch_normalization_2_gammaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp3assignvariableop_23_cdnn_batch_normalization_2_betaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp4assignvariableop_24_cdnn_batch_normalization_3_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp3assignvariableop_25_cdnn_batch_normalization_3_betaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp4assignvariableop_26_cdnn_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp3assignvariableop_27_cdnn_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp4assignvariableop_28_cdnn_batch_normalization_5_gammaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp3assignvariableop_29_cdnn_batch_normalization_5_betaIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp4assignvariableop_30_cdnn_batch_normalization_6_gammaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp3assignvariableop_31_cdnn_batch_normalization_6_betaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp4assignvariableop_32_cdnn_batch_normalization_7_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp3assignvariableop_33_cdnn_batch_normalization_7_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp4assignvariableop_34_cdnn_batch_normalization_8_gammaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp3assignvariableop_35_cdnn_batch_normalization_8_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp8assignvariableop_36_cdnn_batch_normalization_moving_meanIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp<assignvariableop_37_cdnn_batch_normalization_moving_varianceIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp:assignvariableop_38_cdnn_batch_normalization_1_moving_meanIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp>assignvariableop_39_cdnn_batch_normalization_1_moving_varianceIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp:assignvariableop_40_cdnn_batch_normalization_2_moving_meanIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp>assignvariableop_41_cdnn_batch_normalization_2_moving_varianceIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp:assignvariableop_42_cdnn_batch_normalization_3_moving_meanIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp>assignvariableop_43_cdnn_batch_normalization_3_moving_varianceIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp:assignvariableop_44_cdnn_batch_normalization_4_moving_meanIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp>assignvariableop_45_cdnn_batch_normalization_4_moving_varianceIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp:assignvariableop_46_cdnn_batch_normalization_5_moving_meanIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp>assignvariableop_47_cdnn_batch_normalization_5_moving_varianceIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp:assignvariableop_48_cdnn_batch_normalization_6_moving_meanIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp>assignvariableop_49_cdnn_batch_normalization_6_moving_varianceIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp:assignvariableop_50_cdnn_batch_normalization_7_moving_meanIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp>assignvariableop_51_cdnn_batch_normalization_7_moving_varianceIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp:assignvariableop_52_cdnn_batch_normalization_8_moving_meanIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp>assignvariableop_53_cdnn_batch_normalization_8_moving_varianceIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp'assignvariableop_54_cdnn_dense_9_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp%assignvariableop_55_cdnn_dense_9_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_iterationIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp!assignvariableop_57_learning_rateIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp+assignvariableop_58_sgd_m_cdnn_dense_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp)assignvariableop_59_sgd_m_cdnn_dense_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp-assignvariableop_60_sgd_m_cdnn_dense_1_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp+assignvariableop_61_sgd_m_cdnn_dense_1_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp-assignvariableop_62_sgd_m_cdnn_dense_2_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp+assignvariableop_63_sgd_m_cdnn_dense_2_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp-assignvariableop_64_sgd_m_cdnn_dense_3_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp+assignvariableop_65_sgd_m_cdnn_dense_3_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp-assignvariableop_66_sgd_m_cdnn_dense_4_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp+assignvariableop_67_sgd_m_cdnn_dense_4_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp-assignvariableop_68_sgd_m_cdnn_dense_5_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp+assignvariableop_69_sgd_m_cdnn_dense_5_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp-assignvariableop_70_sgd_m_cdnn_dense_6_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp+assignvariableop_71_sgd_m_cdnn_dense_6_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp-assignvariableop_72_sgd_m_cdnn_dense_7_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp+assignvariableop_73_sgd_m_cdnn_dense_7_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp-assignvariableop_74_sgd_m_cdnn_dense_8_kernelIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp+assignvariableop_75_sgd_m_cdnn_dense_8_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp8assignvariableop_76_sgd_m_cdnn_batch_normalization_gammaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp7assignvariableop_77_sgd_m_cdnn_batch_normalization_betaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp:assignvariableop_78_sgd_m_cdnn_batch_normalization_1_gammaIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp9assignvariableop_79_sgd_m_cdnn_batch_normalization_1_betaIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp:assignvariableop_80_sgd_m_cdnn_batch_normalization_2_gammaIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp9assignvariableop_81_sgd_m_cdnn_batch_normalization_2_betaIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp:assignvariableop_82_sgd_m_cdnn_batch_normalization_3_gammaIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp9assignvariableop_83_sgd_m_cdnn_batch_normalization_3_betaIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp:assignvariableop_84_sgd_m_cdnn_batch_normalization_4_gammaIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp9assignvariableop_85_sgd_m_cdnn_batch_normalization_4_betaIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp:assignvariableop_86_sgd_m_cdnn_batch_normalization_5_gammaIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp9assignvariableop_87_sgd_m_cdnn_batch_normalization_5_betaIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp:assignvariableop_88_sgd_m_cdnn_batch_normalization_6_gammaIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp9assignvariableop_89_sgd_m_cdnn_batch_normalization_6_betaIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp:assignvariableop_90_sgd_m_cdnn_batch_normalization_7_gammaIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp9assignvariableop_91_sgd_m_cdnn_batch_normalization_7_betaIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp:assignvariableop_92_sgd_m_cdnn_batch_normalization_8_gammaIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp9assignvariableop_93_sgd_m_cdnn_batch_normalization_8_betaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp-assignvariableop_94_sgd_m_cdnn_dense_9_kernelIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp+assignvariableop_95_sgd_m_cdnn_dense_9_biasIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOpassignvariableop_96_total_1Identity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOpassignvariableop_97_count_1Identity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOpassignvariableop_98_totalIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOpassignvariableop_99_countIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_100Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_101IdentityIdentity_100:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_101Identity_101:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_1088746

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
D__inference_dense_4_layer_call_and_return_conditional_losses_1091298

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
�
�
D__inference_dense_7_layer_call_and_return_conditional_losses_1091370

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_7/kernel/Regularizer/L2LossL2Loss=cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_7/kernel/Regularizer/mulMul.cdnn/dense_7/kernel/Regularizer/mul/x:output:0/cdnn/dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_dense_6_layer_call_and_return_conditional_losses_1091346

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
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
�$
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1088289

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
�$
�
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1088125

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
�
�
)__inference_dense_6_layer_call_fn_1091331

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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1088666o
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
D__inference_dense_5_layer_call_and_return_conditional_losses_1091322

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
�	
�
__inference_loss_fn_5_1091046P
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
5__inference_batch_normalization_layer_call_fn_1091420

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
GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1087797o
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
�
�
&__inference_cdnn_layer_call_fn_1090337
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

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:
identity��StatefulPartitionedCall�
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
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(& #$%&)*+,/0125678*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_cdnn_layer_call_and_return_conditional_losses_1089356o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�
�
D__inference_dense_3_layer_call_and_return_conditional_losses_1088576

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
)__inference_dense_7_layer_call_fn_1091355

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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1088696o
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
�	
�
__inference_loss_fn_9_1091082P
>cdnn_dense_9_kernel_regularizer_l2loss_readvariableop_resource:
identity��5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>cdnn_dense_9_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_9/kernel/Regularizer/L2LossL2Loss=cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_9/kernel/Regularizer/mulMul.cdnn/dense_9/kernel/Regularizer/mul/x:output:0/cdnn/dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'cdnn/dense_9/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp
��
�
A__inference_cdnn_layer_call_and_return_conditional_losses_1088816
p_tinput
dense_1088487:
dense_1088489:)
batch_normalization_1088492:)
batch_normalization_1088494:)
batch_normalization_1088496:)
batch_normalization_1088498:!
dense_1_1088517:
dense_1_1088519:+
batch_normalization_1_1088522:+
batch_normalization_1_1088524:+
batch_normalization_1_1088526:+
batch_normalization_1_1088528:!
dense_2_1088547:
dense_2_1088549:+
batch_normalization_2_1088552:+
batch_normalization_2_1088554:+
batch_normalization_2_1088556:+
batch_normalization_2_1088558:!
dense_3_1088577:
dense_3_1088579:+
batch_normalization_3_1088582:+
batch_normalization_3_1088584:+
batch_normalization_3_1088586:+
batch_normalization_3_1088588:!
dense_4_1088607:
dense_4_1088609:+
batch_normalization_4_1088612:+
batch_normalization_4_1088614:+
batch_normalization_4_1088616:+
batch_normalization_4_1088618:!
dense_5_1088637:
dense_5_1088639:+
batch_normalization_5_1088642:+
batch_normalization_5_1088644:+
batch_normalization_5_1088646:+
batch_normalization_5_1088648:!
dense_6_1088667:
dense_6_1088669:+
batch_normalization_6_1088672:+
batch_normalization_6_1088674:+
batch_normalization_6_1088676:+
batch_normalization_6_1088678:!
dense_7_1088697:
dense_7_1088699:+
batch_normalization_7_1088702:+
batch_normalization_7_1088704:+
batch_normalization_7_1088706:+
batch_normalization_7_1088708:!
dense_8_1088727:
dense_8_1088729:+
batch_normalization_8_1088732:+
batch_normalization_8_1088734:+
batch_normalization_8_1088736:+
batch_normalization_8_1088738:!
dense_9_1088763:
dense_9_1088765:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallp_tinputdense_1088487dense_1088489*
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1088486�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1088492batch_normalization_1088494batch_normalization_1088496batch_normalization_1088498*
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
GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1087750�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_1088517dense_1_1088519*
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1088516�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_1088522batch_normalization_1_1088524batch_normalization_1_1088526batch_normalization_1_1088528*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1087832�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_1088547dense_2_1088549*
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1088546�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_1088552batch_normalization_2_1088554batch_normalization_2_1088556batch_normalization_2_1088558*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1087914�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_1088577dense_3_1088579*
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
GPU 2J 8� *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1088576�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_3_1088582batch_normalization_3_1088584batch_normalization_3_1088586batch_normalization_3_1088588*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1087996�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_1088607dense_4_1088609*
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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1088606�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_4_1088612batch_normalization_4_1088614batch_normalization_4_1088616batch_normalization_4_1088618*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1088078�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_1088637dense_5_1088639*
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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1088636�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_5_1088642batch_normalization_5_1088644batch_normalization_5_1088646batch_normalization_5_1088648*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1088160�
dense_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_6_1088667dense_6_1088669*
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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1088666�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_6_1088672batch_normalization_6_1088674batch_normalization_6_1088676batch_normalization_6_1088678*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1088242�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_7_1088697dense_7_1088699*
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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1088696�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_7_1088702batch_normalization_7_1088704batch_normalization_7_1088706batch_normalization_7_1088708*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1088324�
dense_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_8_1088727dense_8_1088729*
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
GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1088726�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_8_1088732batch_normalization_8_1088734batch_normalization_8_1088736batch_normalization_8_1088738*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1088406�
dropout/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1088746�
dense_9/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_9_1088763dense_9_1088765*
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
GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1088762�
activation/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1088773�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1088487*
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
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_1088517*
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
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_1088547*
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
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_1088577*
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
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_4_1088607*
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
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_5_1088637*
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
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_1088667*
_output_shapes

:*
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
: �
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_7_1088697*
_output_shapes

:*
dtype0�
&cdnn/dense_7/kernel/Regularizer/L2LossL2Loss=cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_7/kernel/Regularizer/mulMul.cdnn/dense_7/kernel/Regularizer/mul/x:output:0/cdnn/dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_8_1088727*
_output_shapes

:*
dtype0�
&cdnn/dense_8/kernel/Regularizer/L2LossL2Loss=cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_8/kernel/Regularizer/mulMul.cdnn/dense_8/kernel/Regularizer/mul/x:output:0/cdnn/dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_9_1088763*
_output_shapes

:*
dtype0�
&cdnn/dense_9/kernel/Regularizer/L2LossL2Loss=cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_9/kernel/Regularizer/mulMul.cdnn/dense_9/kernel/Regularizer/mul/x:output:0/cdnn/dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tInput
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1088242

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
D__inference_dense_3_layer_call_and_return_conditional_losses_1091274

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
�
�
7__inference_batch_normalization_2_layer_call_fn_1091567

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1087914o
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
�
b
)__inference_dropout_layer_call_fn_1091128

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
GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1088967o
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
�
�
D__inference_dense_8_layer_call_and_return_conditional_losses_1091394

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpt
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
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_8/kernel/Regularizer/L2LossL2Loss=cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_8/kernel/Regularizer/mulMul.cdnn/dense_8/kernel/Regularizer/mul/x:output:0/cdnn/dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_1091133

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
�$
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1091714

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
D__inference_dense_1_layer_call_and_return_conditional_losses_1091226

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
�
�
7__inference_batch_normalization_8_layer_call_fn_1092047

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1088406o
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1091520

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1091600

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1087832

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1091920

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
�
�
7__inference_batch_normalization_4_layer_call_fn_1091740

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1088125o
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1091954

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
�
�
)__inference_dense_9_layer_call_fn_1091154

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
GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1088762o
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
�
�
)__inference_dense_4_layer_call_fn_1091283

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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1088606o
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
�
�
)__inference_dense_1_layer_call_fn_1091211

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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1088516o
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
�
�
7__inference_batch_normalization_1_layer_call_fn_1091487

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1087832o
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
��
�
A__inference_cdnn_layer_call_and_return_conditional_losses_1089765
input_1
dense_1089591:
dense_1089593:)
batch_normalization_1089596:)
batch_normalization_1089598:)
batch_normalization_1089600:)
batch_normalization_1089602:!
dense_1_1089605:
dense_1_1089607:+
batch_normalization_1_1089610:+
batch_normalization_1_1089612:+
batch_normalization_1_1089614:+
batch_normalization_1_1089616:!
dense_2_1089619:
dense_2_1089621:+
batch_normalization_2_1089624:+
batch_normalization_2_1089626:+
batch_normalization_2_1089628:+
batch_normalization_2_1089630:!
dense_3_1089633:
dense_3_1089635:+
batch_normalization_3_1089638:+
batch_normalization_3_1089640:+
batch_normalization_3_1089642:+
batch_normalization_3_1089644:!
dense_4_1089647:
dense_4_1089649:+
batch_normalization_4_1089652:+
batch_normalization_4_1089654:+
batch_normalization_4_1089656:+
batch_normalization_4_1089658:!
dense_5_1089661:
dense_5_1089663:+
batch_normalization_5_1089666:+
batch_normalization_5_1089668:+
batch_normalization_5_1089670:+
batch_normalization_5_1089672:!
dense_6_1089675:
dense_6_1089677:+
batch_normalization_6_1089680:+
batch_normalization_6_1089682:+
batch_normalization_6_1089684:+
batch_normalization_6_1089686:!
dense_7_1089689:
dense_7_1089691:+
batch_normalization_7_1089694:+
batch_normalization_7_1089696:+
batch_normalization_7_1089698:+
batch_normalization_7_1089700:!
dense_8_1089703:
dense_8_1089705:+
batch_normalization_8_1089708:+
batch_normalization_8_1089710:+
batch_normalization_8_1089712:+
batch_normalization_8_1089714:!
dense_9_1089718:
dense_9_1089720:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1089591dense_1089593*
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1088486�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1089596batch_normalization_1089598batch_normalization_1089600batch_normalization_1089602*
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
GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1087750�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_1089605dense_1_1089607*
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1088516�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_1089610batch_normalization_1_1089612batch_normalization_1_1089614batch_normalization_1_1089616*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1087832�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_1089619dense_2_1089621*
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1088546�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_1089624batch_normalization_2_1089626batch_normalization_2_1089628batch_normalization_2_1089630*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1087914�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_1089633dense_3_1089635*
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
GPU 2J 8� *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1088576�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_3_1089638batch_normalization_3_1089640batch_normalization_3_1089642batch_normalization_3_1089644*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1087996�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_1089647dense_4_1089649*
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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1088606�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_4_1089652batch_normalization_4_1089654batch_normalization_4_1089656batch_normalization_4_1089658*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1088078�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_1089661dense_5_1089663*
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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1088636�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_5_1089666batch_normalization_5_1089668batch_normalization_5_1089670batch_normalization_5_1089672*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1088160�
dense_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_6_1089675dense_6_1089677*
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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1088666�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_6_1089680batch_normalization_6_1089682batch_normalization_6_1089684batch_normalization_6_1089686*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1088242�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_7_1089689dense_7_1089691*
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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1088696�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_7_1089694batch_normalization_7_1089696batch_normalization_7_1089698batch_normalization_7_1089700*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1088324�
dense_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_8_1089703dense_8_1089705*
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
GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1088726�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_8_1089708batch_normalization_8_1089710batch_normalization_8_1089712batch_normalization_8_1089714*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1088406�
dropout/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1088746�
dense_9/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_9_1089718dense_9_1089720*
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
GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1088762�
activation/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1088773�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1089591*
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
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_1089605*
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
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_1089619*
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
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_1089633*
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
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_4_1089647*
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
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_5_1089661*
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
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_1089675*
_output_shapes

:*
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
: �
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_7_1089689*
_output_shapes

:*
dtype0�
&cdnn/dense_7/kernel/Regularizer/L2LossL2Loss=cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_7/kernel/Regularizer/mulMul.cdnn/dense_7/kernel/Regularizer/mul/x:output:0/cdnn/dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_8_1089703*
_output_shapes

:*
dtype0�
&cdnn/dense_8/kernel/Regularizer/L2LossL2Loss=cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_8/kernel/Regularizer/mulMul.cdnn/dense_8/kernel/Regularizer/mul/x:output:0/cdnn/dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_9_1089718*
_output_shapes

:*
dtype0�
&cdnn/dense_9/kernel/Regularizer/L2LossL2Loss=cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_9/kernel/Regularizer/mulMul.cdnn/dense_9/kernel/Regularizer/mul/x:output:0/cdnn/dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
7__inference_batch_normalization_6_layer_call_fn_1091887

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1088242o
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
c
G__inference_activation_layer_call_and_return_conditional_losses_1091178

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
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1088406

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
E
)__inference_dropout_layer_call_fn_1091123

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
GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1088746`
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
c
G__inference_activation_layer_call_and_return_conditional_losses_1088773

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
7__inference_batch_normalization_5_layer_call_fn_1091820

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1088207o
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
7__inference_batch_normalization_5_layer_call_fn_1091807

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1088160o
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
)__inference_dense_2_layer_call_fn_1091235

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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1088546o
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
ɷ
�
A__inference_cdnn_layer_call_and_return_conditional_losses_1089356
p_tinput
dense_1089182:
dense_1089184:)
batch_normalization_1089187:)
batch_normalization_1089189:)
batch_normalization_1089191:)
batch_normalization_1089193:!
dense_1_1089196:
dense_1_1089198:+
batch_normalization_1_1089201:+
batch_normalization_1_1089203:+
batch_normalization_1_1089205:+
batch_normalization_1_1089207:!
dense_2_1089210:
dense_2_1089212:+
batch_normalization_2_1089215:+
batch_normalization_2_1089217:+
batch_normalization_2_1089219:+
batch_normalization_2_1089221:!
dense_3_1089224:
dense_3_1089226:+
batch_normalization_3_1089229:+
batch_normalization_3_1089231:+
batch_normalization_3_1089233:+
batch_normalization_3_1089235:!
dense_4_1089238:
dense_4_1089240:+
batch_normalization_4_1089243:+
batch_normalization_4_1089245:+
batch_normalization_4_1089247:+
batch_normalization_4_1089249:!
dense_5_1089252:
dense_5_1089254:+
batch_normalization_5_1089257:+
batch_normalization_5_1089259:+
batch_normalization_5_1089261:+
batch_normalization_5_1089263:!
dense_6_1089266:
dense_6_1089268:+
batch_normalization_6_1089271:+
batch_normalization_6_1089273:+
batch_normalization_6_1089275:+
batch_normalization_6_1089277:!
dense_7_1089280:
dense_7_1089282:+
batch_normalization_7_1089285:+
batch_normalization_7_1089287:+
batch_normalization_7_1089289:+
batch_normalization_7_1089291:!
dense_8_1089294:
dense_8_1089296:+
batch_normalization_8_1089299:+
batch_normalization_8_1089301:+
batch_normalization_8_1089303:+
batch_normalization_8_1089305:!
dense_9_1089309:
dense_9_1089311:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallp_tinputdense_1089182dense_1089184*
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1088486�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1089187batch_normalization_1089189batch_normalization_1089191batch_normalization_1089193*
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
GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1087797�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_1089196dense_1_1089198*
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1088516�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_1089201batch_normalization_1_1089203batch_normalization_1_1089205batch_normalization_1_1089207*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1087879�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_1089210dense_2_1089212*
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1088546�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_1089215batch_normalization_2_1089217batch_normalization_2_1089219batch_normalization_2_1089221*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1087961�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_1089224dense_3_1089226*
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
GPU 2J 8� *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1088576�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_3_1089229batch_normalization_3_1089231batch_normalization_3_1089233batch_normalization_3_1089235*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1088043�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_1089238dense_4_1089240*
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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1088606�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_4_1089243batch_normalization_4_1089245batch_normalization_4_1089247batch_normalization_4_1089249*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1088125�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_1089252dense_5_1089254*
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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1088636�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_5_1089257batch_normalization_5_1089259batch_normalization_5_1089261batch_normalization_5_1089263*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1088207�
dense_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_6_1089266dense_6_1089268*
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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1088666�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_6_1089271batch_normalization_6_1089273batch_normalization_6_1089275batch_normalization_6_1089277*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1088289�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_7_1089280dense_7_1089282*
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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1088696�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_7_1089285batch_normalization_7_1089287batch_normalization_7_1089289batch_normalization_7_1089291*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1088371�
dense_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_8_1089294dense_8_1089296*
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
GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1088726�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_8_1089299batch_normalization_8_1089301batch_normalization_8_1089303batch_normalization_8_1089305*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1088453�
dropout/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1088967�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_9_1089309dense_9_1089311*
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
GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1088762�
activation/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1088773�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1089182*
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
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_1089196*
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
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_1089210*
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
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_1089224*
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
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_4_1089238*
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
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_5_1089252*
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
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_1089266*
_output_shapes

:*
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
: �
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_7_1089280*
_output_shapes

:*
dtype0�
&cdnn/dense_7/kernel/Regularizer/L2LossL2Loss=cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_7/kernel/Regularizer/mulMul.cdnn/dense_7/kernel/Regularizer/mul/x:output:0/cdnn/dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_8_1089294*
_output_shapes

:*
dtype0�
&cdnn/dense_8/kernel/Regularizer/L2LossL2Loss=cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_8/kernel/Regularizer/mulMul.cdnn/dense_8/kernel/Regularizer/mul/x:output:0/cdnn/dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_9_1089309*
_output_shapes

:*
dtype0�
&cdnn/dense_9/kernel/Regularizer/L2LossL2Loss=cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_9/kernel/Regularizer/mulMul.cdnn/dense_9/kernel/Regularizer/mul/x:output:0/cdnn/dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tInput
�$
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1088453

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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1092080

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
�
�
7__inference_batch_normalization_6_layer_call_fn_1091900

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1088289o
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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1087797

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
�
�
%__inference_signature_wrapper_1090103
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

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:
identity��StatefulPartitionedCall�
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
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1087726o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
7__inference_batch_normalization_2_layer_call_fn_1091580

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1087961o
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
Ʒ
�
A__inference_cdnn_layer_call_and_return_conditional_losses_1089942
input_1
dense_1089768:
dense_1089770:)
batch_normalization_1089773:)
batch_normalization_1089775:)
batch_normalization_1089777:)
batch_normalization_1089779:!
dense_1_1089782:
dense_1_1089784:+
batch_normalization_1_1089787:+
batch_normalization_1_1089789:+
batch_normalization_1_1089791:+
batch_normalization_1_1089793:!
dense_2_1089796:
dense_2_1089798:+
batch_normalization_2_1089801:+
batch_normalization_2_1089803:+
batch_normalization_2_1089805:+
batch_normalization_2_1089807:!
dense_3_1089810:
dense_3_1089812:+
batch_normalization_3_1089815:+
batch_normalization_3_1089817:+
batch_normalization_3_1089819:+
batch_normalization_3_1089821:!
dense_4_1089824:
dense_4_1089826:+
batch_normalization_4_1089829:+
batch_normalization_4_1089831:+
batch_normalization_4_1089833:+
batch_normalization_4_1089835:!
dense_5_1089838:
dense_5_1089840:+
batch_normalization_5_1089843:+
batch_normalization_5_1089845:+
batch_normalization_5_1089847:+
batch_normalization_5_1089849:!
dense_6_1089852:
dense_6_1089854:+
batch_normalization_6_1089857:+
batch_normalization_6_1089859:+
batch_normalization_6_1089861:+
batch_normalization_6_1089863:!
dense_7_1089866:
dense_7_1089868:+
batch_normalization_7_1089871:+
batch_normalization_7_1089873:+
batch_normalization_7_1089875:+
batch_normalization_7_1089877:!
dense_8_1089880:
dense_8_1089882:+
batch_normalization_8_1089885:+
batch_normalization_8_1089887:+
batch_normalization_8_1089889:+
batch_normalization_8_1089891:!
dense_9_1089895:
dense_9_1089897:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1089768dense_1089770*
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1088486�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1089773batch_normalization_1089775batch_normalization_1089777batch_normalization_1089779*
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
GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1087797�
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_1089782dense_1_1089784*
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1088516�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_1089787batch_normalization_1_1089789batch_normalization_1_1089791batch_normalization_1_1089793*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1087879�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_1089796dense_2_1089798*
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1088546�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_1089801batch_normalization_2_1089803batch_normalization_2_1089805batch_normalization_2_1089807*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1087961�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_1089810dense_3_1089812*
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
GPU 2J 8� *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1088576�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_3_1089815batch_normalization_3_1089817batch_normalization_3_1089819batch_normalization_3_1089821*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1088043�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_1089824dense_4_1089826*
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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1088606�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_4_1089829batch_normalization_4_1089831batch_normalization_4_1089833batch_normalization_4_1089835*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1088125�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_1089838dense_5_1089840*
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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1088636�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_5_1089843batch_normalization_5_1089845batch_normalization_5_1089847batch_normalization_5_1089849*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1088207�
dense_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_6_1089852dense_6_1089854*
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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1088666�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_6_1089857batch_normalization_6_1089859batch_normalization_6_1089861batch_normalization_6_1089863*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1088289�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_7_1089866dense_7_1089868*
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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1088696�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_7_1089871batch_normalization_7_1089873batch_normalization_7_1089875batch_normalization_7_1089877*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1088371�
dense_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_8_1089880dense_8_1089882*
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
GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1088726�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_8_1089885batch_normalization_8_1089887batch_normalization_8_1089889batch_normalization_8_1089891*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1088453�
dropout/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1088967�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_9_1089895dense_9_1089897*
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
GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1088762�
activation/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1088773�
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1089768*
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
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_1089782*
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
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_1089796*
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
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_3_1089810*
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
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_4_1089824*
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
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_5_1089838*
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
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_1089852*
_output_shapes

:*
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
: �
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_7_1089866*
_output_shapes

:*
dtype0�
&cdnn/dense_7/kernel/Regularizer/L2LossL2Loss=cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_7/kernel/Regularizer/mulMul.cdnn/dense_7/kernel/Regularizer/mul/x:output:0/cdnn/dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_8_1089880*
_output_shapes

:*
dtype0�
&cdnn/dense_8/kernel/Regularizer/L2LossL2Loss=cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_8/kernel/Regularizer/mulMul.cdnn/dense_8/kernel/Regularizer/mul/x:output:0/cdnn/dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_9_1089895*
_output_shapes

:*
dtype0�
&cdnn/dense_9/kernel/Regularizer/L2LossL2Loss=cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_9/kernel/Regularizer/mulMul.cdnn/dense_9/kernel/Regularizer/mul/x:output:0/cdnn/dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
B__inference_dense_layer_call_and_return_conditional_losses_1091202

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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1092034

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
�
H
,__inference_activation_layer_call_fn_1091173

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
GPU 2J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1088773`
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
��
�;
A__inference_cdnn_layer_call_and_return_conditional_losses_1090988
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
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:K
=batch_normalization_6_assignmovingavg_readvariableop_resource:M
?batch_normalization_6_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_6_cast_readvariableop_resource:B
4batch_normalization_6_cast_1_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:K
=batch_normalization_7_assignmovingavg_readvariableop_resource:M
?batch_normalization_7_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:K
=batch_normalization_8_assignmovingavg_readvariableop_resource:M
?batch_normalization_8_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�%batch_normalization_4/AssignMovingAvg�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_1�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�%batch_normalization_5/AssignMovingAvg�4batch_normalization_5/AssignMovingAvg/ReadVariableOp�'batch_normalization_5/AssignMovingAvg_1�6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_5/Cast/ReadVariableOp�+batch_normalization_5/Cast_1/ReadVariableOp�%batch_normalization_6/AssignMovingAvg�4batch_normalization_6/AssignMovingAvg/ReadVariableOp�'batch_normalization_6/AssignMovingAvg_1�6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�%batch_normalization_7/AssignMovingAvg�4batch_normalization_7/AssignMovingAvg/ReadVariableOp�'batch_normalization_7/AssignMovingAvg_1�6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�%batch_normalization_8/AssignMovingAvg�4batch_normalization_8/AssignMovingAvg/ReadVariableOp�'batch_normalization_8/AssignMovingAvg_1�6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
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
:����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_6/moments/meanMeandense_6/Relu:activations:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes

:�
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense_6/Relu:activations:03batch_normalization_6/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_6/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*
_output_shapes
:�
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization_6/AssignMovingAvgAssignSubVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_6/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
'batch_normalization_6/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_6/batchnorm/mul_1Muldense_6/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_6/batchnorm/subSub1batch_normalization_6/Cast/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_7/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_7/moments/meanMeandense_7/Relu:activations:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes

:�
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_7/Relu:activations:03batch_normalization_7/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_7/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*
_output_shapes
:�
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization_7/AssignMovingAvgAssignSubVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_7/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
'batch_normalization_7/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_7/batchnorm/mul_1Muldense_7/Relu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_7/batchnorm/subSub1batch_normalization_7/Cast/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_8/MatMulMatMul)batch_normalization_7/batchnorm/add_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_8/moments/meanMeandense_8/Relu:activations:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes

:�
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_8/Relu:activations:03batch_normalization_8/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_8/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*
_output_shapes
:�
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization_8/AssignMovingAvgAssignSubVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_8/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
'batch_normalization_8/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
%batch_normalization_8/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization_8/batchnorm/subSub1batch_normalization_8/Cast/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?�
dropout/dropout/MulMul)batch_normalization_8/batchnorm/add_1:z:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������n
dropout/dropout/ShapeShape)batch_normalization_8/batchnorm/add_1:z:0*
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
 *���>�
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
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_9/MatMulMatMul!dropout/dropout/SelectV2:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
activation/SigmoidSigmoiddense_9/BiasAdd:output:0*
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

:*
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
: �
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_7/kernel/Regularizer/L2LossL2Loss=cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_7/kernel/Regularizer/mulMul.cdnn/dense_7/kernel/Regularizer/mul/x:output:0/cdnn/dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_8/kernel/Regularizer/L2LossL2Loss=cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_8/kernel/Regularizer/mulMul.cdnn/dense_8/kernel/Regularizer/mul/x:output:0/cdnn/dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
&cdnn/dense_9/kernel/Regularizer/L2LossL2Loss=cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%cdnn/dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#cdnn/dense_9/kernel/Regularizer/mulMul.cdnn/dense_9/kernel/Regularizer/mul/x:output:0/cdnn/dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp&^batch_normalization_6/AssignMovingAvg5^batch_normalization_6/AssignMovingAvg/ReadVariableOp(^batch_normalization_6/AssignMovingAvg_17^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp&^batch_normalization_7/AssignMovingAvg5^batch_normalization_7/AssignMovingAvg/ReadVariableOp(^batch_normalization_7/AssignMovingAvg_17^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp&^batch_normalization_8/AssignMovingAvg5^batch_normalization_8/AssignMovingAvg/ReadVariableOp(^batch_normalization_8/AssignMovingAvg_17^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp4^cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp6^cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
+batch_normalization_5/Cast_1/ReadVariableOp+batch_normalization_5/Cast_1/ReadVariableOp2N
%batch_normalization_6/AssignMovingAvg%batch_normalization_6/AssignMovingAvg2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_6/AssignMovingAvg_1'batch_normalization_6/AssignMovingAvg_12p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2N
%batch_normalization_7/AssignMovingAvg%batch_normalization_7/AssignMovingAvg2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_7/AssignMovingAvg_1'batch_normalization_7/AssignMovingAvg_12p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2N
%batch_normalization_8/AssignMovingAvg%batch_normalization_8/AssignMovingAvg2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_8/AssignMovingAvg_1'batch_normalization_8/AssignMovingAvg_12p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2j
3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3cdnn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_4/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_5/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_8/kernel/Regularizer/L2Loss/ReadVariableOp2n
5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp5cdnn/dense_9/kernel/Regularizer/L2Loss/ReadVariableOp2<
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
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
p_tinput
�
�
D__inference_dense_2_layer_call_and_return_conditional_losses_1088546

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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1088078

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
�
�
&__inference_cdnn_layer_call_fn_1088931
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

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:
identity��StatefulPartitionedCall�
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
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_cdnn_layer_call_and_return_conditional_losses_1088816o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�$
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1092114

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
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
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
�
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
637
738
839
940
:41
;42
<43
=44
>45
?46
@47
A48
B49
C50
D51
E52
F53
G54
H55"
trackable_list_wrapper
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
G36
H37"
trackable_list_wrapper
f
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9"
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Xtrace_0
Ytrace_1
Ztrace_2
[trace_32�
&__inference_cdnn_layer_call_fn_1088931
&__inference_cdnn_layer_call_fn_1090220
&__inference_cdnn_layer_call_fn_1090337
&__inference_cdnn_layer_call_fn_1089588�
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
 zXtrace_0zYtrace_1zZtrace_2z[trace_3
�
\trace_0
]trace_1
^trace_2
_trace_32�
A__inference_cdnn_layer_call_and_return_conditional_losses_1090596
A__inference_cdnn_layer_call_and_return_conditional_losses_1090988
A__inference_cdnn_layer_call_and_return_conditional_losses_1089765
A__inference_cdnn_layer_call_and_return_conditional_losses_1089942�
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
 z\trace_0z]trace_1z^trace_2z_trace_3
�B�
"__inference__wrapped_model_1087726input_1"�
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
`DNN.LayerNeurons"
trackable_dict_wrapper
 "
trackable_list_wrapper
_
a0
b1
c2
d3
e4
f5
g6
h7
i8"
trackable_list_wrapper
_
j0
k1
l2
m3
n4
o5
p6
q7
r8"
trackable_list_wrapper
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
y_random_generator"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

Gkernel
Hbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�	momentums
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
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
%:#2cdnn/dense_6/kernel
:2cdnn/dense_6/bias
%:#2cdnn/dense_7/kernel
:2cdnn/dense_7/bias
%:#2cdnn/dense_8/kernel
:2cdnn/dense_8/bias
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
.:,2 cdnn/batch_normalization_6/gamma
-:+2cdnn/batch_normalization_6/beta
.:,2 cdnn/batch_normalization_7/gamma
-:+2cdnn/batch_normalization_7/beta
.:,2 cdnn/batch_normalization_8/gamma
-:+2cdnn/batch_normalization_8/beta
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
6:4 (2&cdnn/batch_normalization_6/moving_mean
::8 (2*cdnn/batch_normalization_6/moving_variance
6:4 (2&cdnn/batch_normalization_7/moving_mean
::8 (2*cdnn/batch_normalization_7/moving_variance
6:4 (2&cdnn/batch_normalization_8/moving_mean
::8 (2*cdnn/batch_normalization_8/moving_variance
%:#2cdnn/dense_9/kernel
:2cdnn/dense_9/bias
�
�trace_02�
__inference_loss_fn_0_1091001�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_1091010�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_1091019�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_1091028�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_1091037�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_1091046�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_1091055�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_1091064�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_8_1091073�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_9_1091082�
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
annotations� *� z�trace_0
�
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17"
trackable_list_wrapper
�
a0
b1
c2
d3
e4
f5
g6
h7
i8
j9
k10
l11
m12
n13
o14
p15
q16
r17
18
19
20"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_cdnn_layer_call_fn_1088931input_1"�
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
&__inference_cdnn_layer_call_fn_1090220p_tinput"�
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
&__inference_cdnn_layer_call_fn_1090337p_tinput"�
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
&__inference_cdnn_layer_call_fn_1089588input_1"�
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
A__inference_cdnn_layer_call_and_return_conditional_losses_1090596p_tinput"�
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
A__inference_cdnn_layer_call_and_return_conditional_losses_1090988p_tinput"�
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
A__inference_cdnn_layer_call_and_return_conditional_losses_1089765input_1"�
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
A__inference_cdnn_layer_call_and_return_conditional_losses_1089942input_1"�
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
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
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

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

!kernel
"bias"
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
5moving_mean
6moving_variance"
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
7moving_mean
8moving_variance"
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
9moving_mean
:moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	)gamma
*beta
;moving_mean
<moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	+gamma
,beta
=moving_mean
>moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	-gamma
.beta
?moving_mean
@moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	/gamma
0beta
Amoving_mean
Bmoving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	1gamma
2beta
Cmoving_mean
Dmoving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	3gamma
4beta
Emoving_mean
Fmoving_variance"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_layer_call_fn_1091123
)__inference_dropout_layer_call_fn_1091128�
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
�
�trace_0
�trace_12�
D__inference_dropout_layer_call_and_return_conditional_losses_1091133
D__inference_dropout_layer_call_and_return_conditional_losses_1091145�
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
_generic_user_object
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
'
R0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_9_layer_call_fn_1091154�
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
D__inference_dense_9_layer_call_and_return_conditional_losses_1091168�
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
 "
trackable_list_wrapper
 "
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
�
�trace_02�
,__inference_activation_layer_call_fn_1091173�
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
G__inference_activation_layer_call_and_return_conditional_losses_1091178�
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
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37"
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
%__inference_signature_wrapper_1090103input_1"�
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
__inference_loss_fn_0_1091001"�
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
__inference_loss_fn_1_1091010"�
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
__inference_loss_fn_2_1091019"�
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
__inference_loss_fn_3_1091028"�
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
__inference_loss_fn_4_1091037"�
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
__inference_loss_fn_5_1091046"�
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
__inference_loss_fn_6_1091055"�
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
__inference_loss_fn_7_1091064"�
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
__inference_loss_fn_8_1091073"�
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
__inference_loss_fn_9_1091082"�
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
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

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
I0"
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
'__inference_dense_layer_call_fn_1091187�
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
B__inference_dense_layer_call_and_return_conditional_losses_1091202�
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
J0"
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
)__inference_dense_1_layer_call_fn_1091211�
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
D__inference_dense_1_layer_call_and_return_conditional_losses_1091226�
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
K0"
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
)__inference_dense_2_layer_call_fn_1091235�
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
D__inference_dense_2_layer_call_and_return_conditional_losses_1091250�
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
L0"
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
)__inference_dense_3_layer_call_fn_1091259�
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
D__inference_dense_3_layer_call_and_return_conditional_losses_1091274�
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
M0"
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
)__inference_dense_4_layer_call_fn_1091283�
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
D__inference_dense_4_layer_call_and_return_conditional_losses_1091298�
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
N0"
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
)__inference_dense_5_layer_call_fn_1091307�
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
D__inference_dense_5_layer_call_and_return_conditional_losses_1091322�
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_6_layer_call_fn_1091331�
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
 z�trace_0
�
�trace_02�
D__inference_dense_6_layer_call_and_return_conditional_losses_1091346�
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
 z�trace_0
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_7_layer_call_fn_1091355�
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
 z�trace_0
�
�trace_02�
D__inference_dense_7_layer_call_and_return_conditional_losses_1091370�
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
 z�trace_0
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_8_layer_call_fn_1091379�
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
 z�trace_0
�
�trace_02�
D__inference_dense_8_layer_call_and_return_conditional_losses_1091394�
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
 z�trace_0
<
#0
$1
52
63"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_batch_normalization_layer_call_fn_1091407
5__inference_batch_normalization_layer_call_fn_1091420�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1091440
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1091474�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
%0
&1
72
83"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_1_layer_call_fn_1091487
7__inference_batch_normalization_1_layer_call_fn_1091500�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1091520
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1091554�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
'0
(1
92
:3"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_2_layer_call_fn_1091567
7__inference_batch_normalization_2_layer_call_fn_1091580�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1091600
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1091634�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
)0
*1
;2
<3"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_3_layer_call_fn_1091647
7__inference_batch_normalization_3_layer_call_fn_1091660�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1091680
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1091714�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
+0
,1
=2
>3"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_4_layer_call_fn_1091727
7__inference_batch_normalization_4_layer_call_fn_1091740�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1091760
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1091794�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
-0
.1
?2
@3"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_5_layer_call_fn_1091807
7__inference_batch_normalization_5_layer_call_fn_1091820�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1091840
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1091874�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
/0
01
A2
B3"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_6_layer_call_fn_1091887
7__inference_batch_normalization_6_layer_call_fn_1091900�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1091920
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1091954�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
10
21
C2
D3"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_7_layer_call_fn_1091967
7__inference_batch_normalization_7_layer_call_fn_1091980�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1092000
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1092034�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
30
41
E2
F3"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_8_layer_call_fn_1092047
7__inference_batch_normalization_8_layer_call_fn_1092060�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1092080
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1092114�
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
 z�trace_0z�trace_1
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
)__inference_dropout_layer_call_fn_1091123inputs"�
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
)__inference_dropout_layer_call_fn_1091128inputs"�
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
D__inference_dropout_layer_call_and_return_conditional_losses_1091133inputs"�
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
D__inference_dropout_layer_call_and_return_conditional_losses_1091145inputs"�
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
R0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_9_layer_call_fn_1091154inputs"�
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
D__inference_dense_9_layer_call_and_return_conditional_losses_1091168inputs"�
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
,__inference_activation_layer_call_fn_1091173inputs"�
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
G__inference_activation_layer_call_and_return_conditional_losses_1091178inputs"�
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
):'2SGD/m/cdnn/dense_6/kernel
#:!2SGD/m/cdnn/dense_6/bias
):'2SGD/m/cdnn/dense_7/kernel
#:!2SGD/m/cdnn/dense_7/bias
):'2SGD/m/cdnn/dense_8/kernel
#:!2SGD/m/cdnn/dense_8/bias
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
2:02&SGD/m/cdnn/batch_normalization_6/gamma
1:/2%SGD/m/cdnn/batch_normalization_6/beta
2:02&SGD/m/cdnn/batch_normalization_7/gamma
1:/2%SGD/m/cdnn/batch_normalization_7/beta
2:02&SGD/m/cdnn/batch_normalization_8/gamma
1:/2%SGD/m/cdnn/batch_normalization_8/beta
):'2SGD/m/cdnn/dense_9/kernel
#:!2SGD/m/cdnn/dense_9/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
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
I0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_layer_call_fn_1091187inputs"�
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
B__inference_dense_layer_call_and_return_conditional_losses_1091202inputs"�
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
J0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_1_layer_call_fn_1091211inputs"�
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
D__inference_dense_1_layer_call_and_return_conditional_losses_1091226inputs"�
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
K0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_2_layer_call_fn_1091235inputs"�
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
D__inference_dense_2_layer_call_and_return_conditional_losses_1091250inputs"�
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
L0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_3_layer_call_fn_1091259inputs"�
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
D__inference_dense_3_layer_call_and_return_conditional_losses_1091274inputs"�
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
M0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_4_layer_call_fn_1091283inputs"�
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
D__inference_dense_4_layer_call_and_return_conditional_losses_1091298inputs"�
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
N0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_5_layer_call_fn_1091307inputs"�
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
D__inference_dense_5_layer_call_and_return_conditional_losses_1091322inputs"�
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
O0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_6_layer_call_fn_1091331inputs"�
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
D__inference_dense_6_layer_call_and_return_conditional_losses_1091346inputs"�
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
P0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_7_layer_call_fn_1091355inputs"�
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
D__inference_dense_7_layer_call_and_return_conditional_losses_1091370inputs"�
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
Q0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_8_layer_call_fn_1091379inputs"�
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
D__inference_dense_8_layer_call_and_return_conditional_losses_1091394inputs"�
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
50
61"
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
5__inference_batch_normalization_layer_call_fn_1091407inputs"�
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
5__inference_batch_normalization_layer_call_fn_1091420inputs"�
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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1091440inputs"�
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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1091474inputs"�
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
70
81"
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
7__inference_batch_normalization_1_layer_call_fn_1091487inputs"�
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
7__inference_batch_normalization_1_layer_call_fn_1091500inputs"�
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1091520inputs"�
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1091554inputs"�
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
90
:1"
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
7__inference_batch_normalization_2_layer_call_fn_1091567inputs"�
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
7__inference_batch_normalization_2_layer_call_fn_1091580inputs"�
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1091600inputs"�
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1091634inputs"�
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
;0
<1"
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
7__inference_batch_normalization_3_layer_call_fn_1091647inputs"�
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
7__inference_batch_normalization_3_layer_call_fn_1091660inputs"�
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1091680inputs"�
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1091714inputs"�
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
=0
>1"
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
7__inference_batch_normalization_4_layer_call_fn_1091727inputs"�
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
7__inference_batch_normalization_4_layer_call_fn_1091740inputs"�
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1091760inputs"�
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1091794inputs"�
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
?0
@1"
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
7__inference_batch_normalization_5_layer_call_fn_1091807inputs"�
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
7__inference_batch_normalization_5_layer_call_fn_1091820inputs"�
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1091840inputs"�
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1091874inputs"�
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
A0
B1"
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
7__inference_batch_normalization_6_layer_call_fn_1091887inputs"�
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
7__inference_batch_normalization_6_layer_call_fn_1091900inputs"�
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1091920inputs"�
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1091954inputs"�
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
C0
D1"
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
7__inference_batch_normalization_7_layer_call_fn_1091967inputs"�
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
7__inference_batch_normalization_7_layer_call_fn_1091980inputs"�
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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1092000inputs"�
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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1092034inputs"�
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
E0
F1"
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
7__inference_batch_normalization_8_layer_call_fn_1092047inputs"�
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
7__inference_batch_normalization_8_layer_call_fn_1092060inputs"�
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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1092080inputs"�
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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1092114inputs"�
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
"__inference__wrapped_model_1087726�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GH0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
G__inference_activation_layer_call_and_return_conditional_losses_1091178_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
,__inference_activation_layer_call_fn_1091173T/�,
%�"
 �
inputs���������
� "!�
unknown����������
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1091520i78&%3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1091554i78&%3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
7__inference_batch_normalization_1_layer_call_fn_1091487^78&%3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
7__inference_batch_normalization_1_layer_call_fn_1091500^78&%3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1091600i9:('3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1091634i9:('3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
7__inference_batch_normalization_2_layer_call_fn_1091567^9:('3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
7__inference_batch_normalization_2_layer_call_fn_1091580^9:('3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1091680i;<*)3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1091714i;<*)3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
7__inference_batch_normalization_3_layer_call_fn_1091647^;<*)3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
7__inference_batch_normalization_3_layer_call_fn_1091660^;<*)3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1091760i=>,+3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1091794i=>,+3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
7__inference_batch_normalization_4_layer_call_fn_1091727^=>,+3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
7__inference_batch_normalization_4_layer_call_fn_1091740^=>,+3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1091840i?@.-3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1091874i?@.-3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
7__inference_batch_normalization_5_layer_call_fn_1091807^?@.-3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
7__inference_batch_normalization_5_layer_call_fn_1091820^?@.-3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1091920iAB0/3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1091954iAB0/3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
7__inference_batch_normalization_6_layer_call_fn_1091887^AB0/3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
7__inference_batch_normalization_6_layer_call_fn_1091900^AB0/3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1092000iCD213�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1092034iCD213�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
7__inference_batch_normalization_7_layer_call_fn_1091967^CD213�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
7__inference_batch_normalization_7_layer_call_fn_1091980^CD213�0
)�&
 �
inputs���������
p
� "!�
unknown����������
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1092080iEF433�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1092114iEF433�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
7__inference_batch_normalization_8_layer_call_fn_1092047^EF433�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
7__inference_batch_normalization_8_layer_call_fn_1092060^EF433�0
)�&
 �
inputs���������
p
� "!�
unknown����������
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1091440i56$#3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1091474i56$#3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
5__inference_batch_normalization_layer_call_fn_1091407^56$#3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
5__inference_batch_normalization_layer_call_fn_1091420^56$#3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
A__inference_cdnn_layer_call_and_return_conditional_losses_1089765�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GH@�=
&�#
!�
input_1���������
�

trainingp ",�)
"�
tensor_0���������
� �
A__inference_cdnn_layer_call_and_return_conditional_losses_1089942�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GH@�=
&�#
!�
input_1���������
�

trainingp",�)
"�
tensor_0���������
� �
A__inference_cdnn_layer_call_and_return_conditional_losses_1090596�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GHA�>
'�$
"�
p_tinput���������
�

trainingp ",�)
"�
tensor_0���������
� �
A__inference_cdnn_layer_call_and_return_conditional_losses_1090988�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GHA�>
'�$
"�
p_tinput���������
�

trainingp",�)
"�
tensor_0���������
� �
&__inference_cdnn_layer_call_fn_1088931�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GH@�=
&�#
!�
input_1���������
�

trainingp "!�
unknown����������
&__inference_cdnn_layer_call_fn_1089588�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GH@�=
&�#
!�
input_1���������
�

trainingp"!�
unknown����������
&__inference_cdnn_layer_call_fn_1090220�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GHA�>
'�$
"�
p_tinput���������
�

trainingp "!�
unknown����������
&__inference_cdnn_layer_call_fn_1090337�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GHA�>
'�$
"�
p_tinput���������
�

trainingp"!�
unknown����������
D__inference_dense_1_layer_call_and_return_conditional_losses_1091226c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_1_layer_call_fn_1091211X/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_2_layer_call_and_return_conditional_losses_1091250c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_2_layer_call_fn_1091235X/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_3_layer_call_and_return_conditional_losses_1091274c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_3_layer_call_fn_1091259X/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_4_layer_call_and_return_conditional_losses_1091298c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_4_layer_call_fn_1091283X/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_5_layer_call_and_return_conditional_losses_1091322c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_5_layer_call_fn_1091307X/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_6_layer_call_and_return_conditional_losses_1091346c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_6_layer_call_fn_1091331X/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_7_layer_call_and_return_conditional_losses_1091370c /�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_7_layer_call_fn_1091355X /�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_8_layer_call_and_return_conditional_losses_1091394c!"/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_8_layer_call_fn_1091379X!"/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_9_layer_call_and_return_conditional_losses_1091168cGH/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_9_layer_call_fn_1091154XGH/�,
%�"
 �
inputs���������
� "!�
unknown����������
B__inference_dense_layer_call_and_return_conditional_losses_1091202c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
'__inference_dense_layer_call_fn_1091187X/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dropout_layer_call_and_return_conditional_losses_1091133c3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
D__inference_dropout_layer_call_and_return_conditional_losses_1091145c3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
)__inference_dropout_layer_call_fn_1091123X3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
)__inference_dropout_layer_call_fn_1091128X3�0
)�&
 �
inputs���������
p
� "!�
unknown���������E
__inference_loss_fn_0_1091001$�

� 
� "�
unknown E
__inference_loss_fn_1_1091010$�

� 
� "�
unknown E
__inference_loss_fn_2_1091019$�

� 
� "�
unknown E
__inference_loss_fn_3_1091028$�

� 
� "�
unknown E
__inference_loss_fn_4_1091037$�

� 
� "�
unknown E
__inference_loss_fn_5_1091046$�

� 
� "�
unknown E
__inference_loss_fn_6_1091055$�

� 
� "�
unknown E
__inference_loss_fn_7_1091064$�

� 
� "�
unknown E
__inference_loss_fn_8_1091073$!�

� 
� "�
unknown E
__inference_loss_fn_9_1091082$G�

� 
� "�
unknown �
%__inference_signature_wrapper_1090103�856$#78&%9:(';<*)=>,+?@.-AB0/ CD21!"EF43GH;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������