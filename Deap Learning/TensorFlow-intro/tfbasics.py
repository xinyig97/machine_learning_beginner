import tensorflow as tf 


# computation graph
x1 = tf.constant(5)
x2 = tf.constant(6)

# only define a model to do the multiplication
result = tf.multiply(x1,x2) # output as a tensor instead of performing the action
print(result)

# actually execute the code in a session, which contains open, run, and close

# ----- manually open and close --------
# sess = tf.Session()
# print(sess.run(result))
# sess.close()

# ----- automaticlly close -----------
with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output)