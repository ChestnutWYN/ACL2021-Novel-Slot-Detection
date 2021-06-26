# Novel Slot Detection: A Benchmark for Discovering Unknown Slot Types in the Task-Oriented Dialogue System
Code for paper "Novel Slot Detection: A Benchmark for Discovering Unknown Slot Types in the Task-Oriented Dialogue System." will be released soon

# Issue
Q：**There are two training objectives mentioned in Section 4.1: multiple classifier and binary classifier. But if we use binary classifier, how can we get the ind category? And how to get the results of MSP + binary and GDA + binary?**

A：As we mention in Section4.1—— "In the test stage, for in-domain prediction, we both use the multiple classifier. While, for novel slot detection, we use the multiple classifier or the binary classifier, or both of them". It means binary classifier won't be used for gaining the fine in-domain labels, but for detecting whether a token is a novel slot, and if yes, we will override the fine in-domain labels gained by multiple classifier.

