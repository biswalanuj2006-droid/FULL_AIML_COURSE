"""
AI/ML Engineering Course - Complete Diagram Generator
Generates actual JPG/PNG images for all course categories
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Base path
BASE = 'AI_ML_ENGINEERING_COURSE/diagrams'

def ensure_dirs():
    for d in ['ml', 'dl', 'nlp', 'transformers', 'rag', 'backend', 'math']:
        os.makedirs(f'{BASE}/{d}', exist_ok=True)

# ============================================================
# ML DIAGRAMS
# ============================================================

def ml_workflow():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('Machine Learning Workflow', fontsize=18, fontweight='bold')
    steps = [(1.2,3,'Data\nCollection','#E3F2FD'),(3.4,3,'Preprocessing','#C8E6C9'),
             (5.6,3,'Feature\nEngineering','#FFF3E0'),(7.8,3,'Model\nTraining','#F3E5F5'),
             (10,3,'Evaluation','#FFEBEE'),(12.2,3,'Deployment','#E0F7FA'),
             (14.4,3,'Monitoring','#F1F8E9')]
    for i,(x,y,t,c) in enumerate(steps):
        box = FancyBboxPatch((x-0.9,y-0.6),1.8,1.2,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        ax.add_patch(box); ax.text(x,y,t,fontsize=10,ha='center',va='center',fontweight='bold')
        if i<len(steps)-1: ax.annotate('',xy=(steps[i+1][0]-0.9,y),xytext=(x+0.9,y),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    plt.tight_layout(); plt.savefig(f'{BASE}/ml/ml_workflow.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] ml/ml_workflow.jpg')

def ml_algorithm_comparison():
    fig, ax = plt.subplots(figsize=(12, 7))
    models = ['LR','KNN','DT','RF','GBM','SVM','MLP']
    accuracy = [0.82,0.85,0.78,0.88,0.87,0.84,0.86]
    colors = ['#4CAF50','#2196F3','#FF9800','#9C27B0','#F44336','#00BCD4','#E91E63']
    bars = ax.bar(models, accuracy, color=colors, edgecolor='black')
    for bar, v in zip(bars, accuracy): ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12); ax.set_title('ML Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1); ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(f'{BASE}/ml/model_comparison.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] ml/model_comparison.jpg')

def ml_overfitting():
    fig, ax = plt.subplots(figsize=(10, 6))
    complexity = np.linspace(1, 10, 100)
    train_error = 0.8 * np.exp(-0.3*complexity) + 0.05
    val_error = 0.3*np.exp(0.15*complexity) + 0.2
    ax.plot(complexity, train_error, 'b-', lw=2, label='Training Error')
    ax.plot(complexity, val_error, 'r-', lw=2, label='Validation Error')
    ax.axvline(x=4, color='green', linestyle='--', alpha=0.7, label='Optimal Complexity')
    ax.fill_between(complexity, 0, 0.15, alpha=0.1, color='blue')
    ax.fill_between(complexity, 0.8, 1, alpha=0.1, color='red')
    ax.text(2, 0.08, 'Underfitting', fontsize=12, color='blue', fontweight='bold')
    ax.text(8, 0.85, 'Overfitting', fontsize=12, color='red', fontweight='bold')
    ax.set_xlabel('Model Complexity', fontsize=12); ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{BASE}/ml/overfitting.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] ml/overfitting.jpg')

def ml_confusion_matrix():
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = np.array([[85, 15], [10, 90]])
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=16, fontweight='bold', color='white' if cm[i,j]>50 else 'black')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Predicted 0','Predicted 1'], fontsize=11)
    ax.set_yticklabels(['Actual 0','Actual 1'], fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{BASE}/ml/confusion_matrix.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] ml/confusion_matrix.jpg')

def ml_roc_curve():
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr = np.array([0,0.05,0.1,0.2,0.3,0.5,0.7,1.0])
    tpr = np.array([0,0.4,0.6,0.8,0.88,0.94,0.98,1.0])
    auc_val = np.trapz(tpr, fpr)
    ax.plot(fpr, tpr, 'b-', lw=3, label=f'ROC (AUC={auc_val:.3f})')
    ax.plot([0,1],[0,1], 'k--', lw=2, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.2, color='blue')
    ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{BASE}/ml/roc_curve.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] ml/roc_curve.jpg')

def ml_decision_boundary():
    fig, ax = plt.subplots(figsize=(8, 6))
    np.random.seed(42)
    x0 = np.random.randn(50,2)*1.5+[2,2]; x1 = np.random.randn(50,2)*1.5+[6,6]
    ax.scatter(x0[:,0],x0[:,1],c='blue',label='Class 0',alpha=0.6,edgecolors='black')
    ax.scatter(x1[:,0],x1[:,1],c='red',label='Class 1',alpha=0.6,edgecolors='black')
    xx,yy = np.meshgrid(np.linspace(0,8,100),np.linspace(0,8,100))
    Z = (xx-4)+(yy-4); ax.contour(xx,yy,Z,levels=[0],colors='black',linewidths=2)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{BASE}/ml/decision_boundary.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] ml/decision_boundary.jpg')

def ml_cross_validation():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis('off')
    ax.set_title('K-Fold Cross Validation (K=5)', fontsize=16, fontweight='bold')
    colors = ['#FFCDD2','#C8E6C9','#BBDEFB','#FFF3E0','#F3E5F5']
    for fold in range(5):
        y = 4 - fold*0.8
        for i in range(5):
            x = 1+i*2; c = colors[fold] if i==fold else '#E0E0E0'
            rect = plt.Rectangle((x,y),1.6,0.5,facecolor=c,edgecolor='#333',linewidth=1)
            ax.add_patch(rect)
            ax.text(x+0.8,y+0.25,'Val' if i==fold else 'Train',fontsize=8,ha='center',va='center',fontweight='bold' if i==fold else 'normal')
    ax.text(6,0.5,'Average score across all folds',fontsize=12,ha='center',style='italic')
    plt.tight_layout(); plt.savefig(f'{BASE}/ml/cross_validation.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] ml/cross_validation.jpg')

# ============================================================
# DL DIAGRAMS
# ============================================================

def dl_neural_network():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=16, fontweight='bold')
    layers = {'Input':[6.5,5,3.5],'Hidden':[4.5,6,5,3.5,2.5],'Output':[5]}
    colors_l = ['#E3F2FD','#C8E6C9','#F3E5F5']
    x_pos = [1.5,5,8.5]
    names = ['Input\nLayer','Hidden\nLayer','Output\nLayer']
    neurons_list = [[6.5,5,3.5],[4.5,6,5,3.5,2.5],[5]]
    for i in range(len(x_pos)-1):
        for n1 in neurons_list[i]:
            for n2 in neurons_list[i+1]:
                ax.plot([x_pos[i],x_pos[i+1]],[n1,n2],'gray',alpha=0.3,lw=0.5)
    for i,(neurons,color) in enumerate(zip(neurons_list,colors_l)):
        for y in neurons:
            circle = plt.Circle((x_pos[i],y),0.3,facecolor=color,edgecolor='#333',lw=2,zorder=10)
            ax.add_patch(circle)
        ax.text(x_pos[i],1.5,names[i],fontsize=11,ha='center',fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{BASE}/dl/neural_network.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] dl/neural_network.jpg')

def dl_activation_functions():
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    x = np.linspace(-5,5,200)
    # Sigmoid
    axes[0,0].plot(x,1/(1+np.exp(-x)),'b-',lw=3); axes[0,0].set_title('Sigmoid',fontsize=13,fontweight='bold')
    axes[0,0].axhline(y=0.5,color='r',ls='--',alpha=0.5); axes[0,0].grid(True,alpha=0.3)
    # Tanh
    axes[0,1].plot(x,np.tanh(x),'g-',lw=3); axes[0,1].set_title('Tanh',fontsize=13,fontweight='bold')
    axes[0,1].axhline(y=0,color='r',ls='--',alpha=0.5); axes[0,1].grid(True,alpha=0.3)
    # ReLU
    axes[1,0].plot(x,np.maximum(0,x),'r-',lw=3); axes[1,0].set_title('ReLU',fontsize=13,fontweight='bold')
    axes[1,0].axhline(y=0,color='gray',ls='--',alpha=0.5); axes[1,0].grid(True,alpha=0.3)
    # Softmax (conceptual)
    z = np.array([2.0,1.0,0.5]); sm = np.exp(z)/np.exp(z).sum()
    axes[1,1].bar(['Class 0','Class 1','Class 2'],sm,color=['#4CAF50','#2196F3','#FF9800'],edgecolor='black')
    axes[1,1].set_title('Softmax',fontsize=13,fontweight='bold'); axes[1,1].set_ylim(0,1); axes[1,1].grid(True,alpha=0.3,axis='y')
    plt.suptitle('Activation Functions',fontsize=16,fontweight='bold',y=1.01)
    plt.tight_layout(); plt.savefig(f'{BASE}/dl/activation_functions.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] dl/activation_functions.jpg')

def dl_backpropagation():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0,14); ax.set_ylim(0,5); ax.axis('off')
    ax.set_title('Backpropagation Flow', fontsize=16, fontweight='bold')
    steps = [(1.5,2.5,'Input\nData','#E3F2FD'),(4,2.5,'Forward\nPass','#C8E6C9'),
             (6.5,2.5,'Loss\nCalculation','#FFF3E0'),(9,2.5,'Backward\nPass','#FFCDD2'),
             (11.5,2.5,'Weight\nUpdate','#F3E5F5')]
    for i,(x,y,t,c) in enumerate(steps):
        box = FancyBboxPatch((x-1,y-0.7),2,1.4,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        ax.add_patch(box); ax.text(x,y,t,fontsize=11,ha='center',va='center',fontweight='bold')
        if i<len(steps)-1: ax.annotate('',xy=(steps[i+1][0]-1,y),xytext=(x+1,y),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    # Loop arrow
    ax.annotate('Repeat',xy=(1.5,1),xytext=(11.5,1),arrowprops=dict(arrowstyle='->',color='green',lw=2,connectionstyle='arc3,rad=0.3'),fontsize=12,color='green',fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{BASE}/dl/backpropagation.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] dl/backpropagation.jpg')

def dl_cnn_architecture():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0,14); ax.set_ylim(0,5); ax.axis('off')
    ax.set_title('CNN Architecture', fontsize=16, fontweight='bold')
    layers_cnn = [(1.5,2.5,'Input\nImage','#E3F2FD',2),(4,2.5,'Conv\nLayer','#C8E6C9',1.5),
                  (6,2.5,'Pooling\nLayer','#FFF3E0',1.2),(8,2.5,'Conv\nLayer','#F3E5F5',1),
                  (10,2.5,'Flatten\n+ Dense','#FFEBEE',0.8),(12,2.5,'Output','#E0F7FA',0.5)]
    for x,y,t,c,w in layers_cnn:
        box = FancyBboxPatch((x-w/2,y-0.7),w,1.4,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        ax.add_patch(box); ax.text(x,y,t,fontsize=9,ha='center',va='center',fontweight='bold')
    for i in range(len(layers_cnn)-1):
        ax.annotate('',xy=(layers_cnn[i+1][0]-layers_cnn[i+1][4]/2,2.5),xytext=(layers_cnn[i][0]+layers_cnn[i][4]/2,2.5),
                   arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    plt.tight_layout(); plt.savefig(f'{BASE}/dl/cnn_architecture.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] dl/cnn_architecture.jpg')

def dl_lstm_gates():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0,14); ax.set_ylim(0,6); ax.axis('off')
    ax.set_title('LSTM Cell Architecture', fontsize=16, fontweight='bold')
    # Main cell
    cell = FancyBboxPatch((3,1.5),8,3,boxstyle="round,pad=0.2",facecolor='#E8F5E9',edgecolor='#333',linewidth=2)
    ax.add_patch(cell)
    # Gates
    gates = [(4.5,3.5,'Forget\nGate','#FFCDD2'),(7,3.5,'Input\nGate','#C8E6C9'),(9.5,3.5,'Output\nGate','#BBDEFB')]
    for x,y,t,c in gates:
        box = FancyBboxPatch((x-0.7,y-0.5),1.4,1,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=1.5)
        ax.add_patch(box); ax.text(x,y,t,fontsize=9,ha='center',va='center',fontweight='bold')
    ax.text(7,2,'Cell State',fontsize=12,ha='center',fontweight='bold',color='#2E7D32')
    ax.annotate('h(t-1)',xy=(2.5,3),fontsize=11,fontweight='bold')
    ax.annotate('x(t)',xy=(2.5,2),fontsize=11,fontweight='bold')
    ax.annotate('h(t)',xy=(12,3),fontsize=11,fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{BASE}/dl/lstm_gates.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] dl/lstm_gates.jpg')

def dl_gradient_descent():
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    x = np.linspace(0,10,100)
    # Batch GD
    axes[0].plot(x,(x-2)**2,'b-',lw=2); axes[0].plot(8,(8-2)**2,'ro',ms=10); axes[0].plot(2,0,'g*',ms=15)
    axes[0].annotate('',xy=(5,9),xytext=(8,36),arrowprops=dict(arrowstyle='->',color='red',lw=2))
    axes[0].set_title('Batch GD',fontsize=13,fontweight='bold'); axes[0].grid(True,alpha=0.3)
    # Stochastic GD
    axes[1].plot(x,(x-2)**2,'b-',lw=2)
    points = [8,6,5,3.5,2.5,2.1]; prev = points[0]
    for p in points[1:]: axes[1].annotate('',xy=(p,(p-2)**2),xytext=(prev,(prev-2)**2),arrowprops=dict(arrowstyle='->',color='red',lw=1.5)); prev=p
    axes[1].plot(2,0,'g*',ms=15); axes[1].set_title('Stochastic GD',fontsize=13,fontweight='bold'); axes[1].grid(True,alpha=0.3)
    # Mini-batch GD
    axes[2].plot(x,(x-2)**2,'b-',lw=2)
    pts2 = [8,5,3,2.2]; prev2 = pts2[0]
    for p in pts2[1:]: axes[2].annotate('',xy=(p,(p-2)**2),xytext=(prev2,(prev2-2)**2),arrowprops=dict(arrowstyle='->',color='red',lw=2)); prev2=p
    axes[2].plot(2,0,'g*',ms=15); axes[2].set_title('Mini-Batch GD',fontsize=13,fontweight='bold'); axes[2].grid(True,alpha=0.3)
    plt.suptitle('Gradient Descent Variants',fontsize=16,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(f'{BASE}/dl/gradient_descent.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] dl/gradient_descent.jpg')

# ============================================================
# NLP DIAGRAMS
# ============================================================

def nlp_pipeline():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0,16); ax.set_ylim(0,5); ax.axis('off')
    ax.set_title('NLP Pipeline', fontsize=16, fontweight='bold')
    steps = [(1.2,2.5,'Raw\nText','#E3F2FD'),(3.4,2.5,'Token-\nization','#C8E6C9'),
             (5.6,2.5,'Cleaning','#FFF3E0'),(7.8,2.5,'Embedding','#F3E5F5'),
             (10,2.5,'Model','#FFEBEE'),(12.2,2.5,'Prediction','#E0F7FA'),
             (14.4,2.5,'Output','#F1F8E9')]
    for i,(x,y,t,c) in enumerate(steps):
        box = FancyBboxPatch((x-0.9,y-0.6),1.8,1.2,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        ax.add_patch(box); ax.text(x,y,t,fontsize=10,ha='center',va='center',fontweight='bold')
        if i<len(steps)-1: ax.annotate('',xy=(steps[i+1][0]-0.9,y),xytext=(x+0.9,y),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    plt.tight_layout(); plt.savefig(f'{BASE}/nlp/nlp_pipeline.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] nlp/nlp_pipeline.jpg')

def nlp_word_embeddings():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.axis('off')
    ax.set_title('Word Embeddings Space', fontsize=14, fontweight='bold')
    words = {'king':(1.5,2),'queen':(1.8,1.5),'man':(-1,1.5),'woman':(-0.8,0.8),
             'cat':(-2,-1),'dog':(-1.5,-1.5),'car':(0,-2),'bus':(0.5,-2.2)}
    for w,(x,y) in words.items(): ax.plot(x,y,'o',ms=15,markeredgecolor='black'); ax.text(x+0.15,y+0.15,w,fontsize=11,fontweight='bold')
    ax.annotate('',xy=(1.8,1.5),xytext=(1.5,2),arrowprops=dict(arrowstyle='<->',color='green',lw=2))
    ax.text(1.9,1.8,'similar',fontsize=9,color='green')
    ax.annotate('',xy=(-0.8,0.8),xytext=(-1,1.5),arrowprops=dict(arrowstyle='<->',color='blue',lw=2))
    ax.text(-1.2,1,'similar',fontsize=9,color='blue')
    plt.tight_layout(); plt.savefig(f'{BASE}/nlp/word_embeddings.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] nlp/word_embeddings.jpg')

def nlp_sentiment():
    fig, ax = plt.subplots(figsize=(10, 6))
    words = ['amazing','great','good','okay','bad','terrible','awful']
    scores = [0.95,0.8,0.6,0.5,0.3,0.1,0.05]
    colors = ['#4CAF50','#66BB6A','#8BC34A','#FFC107','#FF9800','#F44336','#D32F2F']
    bars = ax.barh(words, scores, color=colors, edgecolor='black')
    ax.set_xlabel('Sentiment Score', fontsize=12); ax.set_title('Sentiment Analysis', fontsize=14, fontweight='bold')
    ax.set_xlim(0,1.1); ax.axvline(x=0.5,color='gray',ls='--',alpha=0.5)
    for bar, v in zip(bars, scores): ax.text(v+0.02, bar.get_y()+bar.get_height()/2, f'{v:.2f}', va='center', fontsize=10)
    plt.tight_layout(); plt.savefig(f'{BASE}/nlp/sentiment_analysis.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] nlp/sentiment_analysis.jpg')

# ============================================================
# TRANSFORMER DIAGRAMS
# ============================================================

def transformer_architecture():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0,12); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title('Transformer Architecture', fontsize=16, fontweight='bold')
    # Encoder side
    enc_box = FancyBboxPatch((0.5,1),5,8,boxstyle="round,pad=0.2",facecolor='#E3F2FD',edgecolor='#1976D2',linewidth=2)
    ax.add_patch(enc_box); ax.text(3,9,'Encoder',fontsize=14,ha='center',fontweight='bold',color='#1976D2')
    enc_layers = [(3,7.5,'Multi-Head\nAttention','#BBDEFB'),(3,5.5,'Add &\nNorm','#FFF9C4'),(3,4,'Feed\nForward','#C8E6C9'),(3,2.5,'Add &\nNorm','#FFF9C4')]
    for x,y,t,c in enc_layers:
        box = FancyBboxPatch((x-1.2,y-0.5),2.4,1,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=1.5)
        ax.add_patch(box); ax.text(x,y,t,fontsize=9,ha='center',va='center',fontweight='bold')
    # Decoder side
    dec_box = FancyBboxPatch((6.5,1),5,8,boxstyle="round,pad=0.2",facecolor='#F3E5F5',edgecolor='#7B1FA2',linewidth=2)
    ax.add_patch(dec_box); ax.text(9,9,'Decoder',fontsize=14,ha='center',fontweight='bold',color='#7B1FA2')
    dec_layers = [(9,7.5,'Masked\nMulti-Head','#E1BEE7'),(9,6,'Add &\nNorm','#FFF9C4'),
                  (9,4.5,'Multi-Head\nAttention','#BBDEFB'),(9,3,'Add &\nNorm','#FFF9C4'),
                  (9,1.8,'Feed\nForward','#C8E6C9')]
    for x,y,t,c in dec_layers:
        box = FancyBboxPatch((x-1.2,y-0.4),2.4,0.8,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=1.5)
        ax.add_patch(box); ax.text(x,y,t,fontsize=8,ha='center',va='center',fontweight='bold')
    # Arrows
    ax.annotate('',xy=(6.5,7.5),xytext=(5.5,7.5),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    ax.annotate('',xy=(6.5,4.5),xytext=(5.5,4.5),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    ax.text(3,0.5,'Input\nEmbeddings',fontsize=10,ha='center',fontweight='bold')
    ax.text(9,0.5,'Output\nEmbeddings',fontsize=10,ha='center',fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{BASE}/transformers/transformer_architecture.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] transformers/transformer_architecture.jpg')

def transformer_attention():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis('off')
    ax.set_title('Self-Attention Mechanism', fontsize=16, fontweight='bold')
    # Q K V
    qkv = [('Q',2,6.5,'#FFCDD2'),('K',5,6.5,'#C8E6C9'),('V',8,6.5,'#BBDEFB')]
    for label,x,y,c in qkv:
        box = FancyBboxPatch((x-0.8,y-0.4),1.6,0.8,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        ax.add_patch(box); ax.text(x,y,label,fontsize=14,ha='center',va='center',fontweight='bold')
    # Score
    ax.annotate('',xy=(3.5,5),xytext=(2,6.1),arrowprops=dict(arrowstyle='->',color='#333',lw=1.5))
    ax.annotate('',xy=(3.5,5),xytext=(5,6.1),arrowprops=dict(arrowstyle='->',color='#333',lw=1.5))
    box = FancyBboxPatch((2.5,4.5),2,1,boxstyle="round,pad=0.1",facecolor='#FFF9C4',edgecolor='#333',linewidth=2)
    ax.add_patch(box); ax.text(3.5,5,'Q x K^T',fontsize=12,ha='center',va='center',fontweight='bold')
    # Scale
    ax.annotate('',xy=(3.5,3.5),xytext=(3.5,4.5),arrowprops=dict(arrowstyle='->',color='#333',lw=1.5))
    box2 = FancyBboxPatch((2.5,3),2,0.8,boxstyle="round,pad=0.1",facecolor='#FFE0B2',edgecolor='#333',linewidth=2)
    ax.add_patch(box2); ax.text(3.5,3.4,'Scale + Softmax',fontsize=10,ha='center',va='center',fontweight='bold')
    # Multiply with V
    ax.annotate('',xy=(6.5,2),xytext=(3.5,3),arrowprops=dict(arrowstyle='->',color='#333',lw=1.5))
    ax.annotate('',xy=(6.5,2),xytext=(8,6.1),arrowprops=dict(arrowstyle='->',color='#333',lw=1.5))
    box3 = FancyBboxPatch((5.5,1.5),2,1,boxstyle="round,pad=0.1",facecolor='#E8F5E9',edgecolor='#333',linewidth=2)
    ax.add_patch(box3); ax.text(6.5,2,'x V',fontsize=12,ha='center',va='center',fontweight='bold')
    # Output
    ax.annotate('',xy=(6.5,0.5),xytext=(6.5,1.5),arrowprops=dict(arrowstyle='->',color='#333',lw=1.5))
    ax.text(6.5,0.2,'Output',fontsize=12,ha='center',fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{BASE}/transformers/self_attention.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] transformers/self_attention.jpg')

# ============================================================
# RAG DIAGRAMS
# ============================================================

def rag_pipeline():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0,16); ax.set_ylim(0,6); ax.axis('off')
    ax.set_title('RAG Pipeline', fontsize=16, fontweight='bold')
    steps = [(1,3,'Documents','#E3F2FD'),(3,3,'Chunking','#C8E6C9'),
             (5,3,'Embedding','#FFF3E0'),(7,3,'Vector\nDB','#F3E5F5'),
             (9,3,'Retrieval','#FFEBEE'),(11,3,'Context','#E0F7FA'),
             (13,3,'LLM','#F1F8E9'),(15,3,'Answer','#FFECB3')]
    for i,(x,y,t,c) in enumerate(steps):
        box = FancyBboxPatch((x-0.8,y-0.6),1.6,1.2,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        ax.add_patch(box); ax.text(x,y,t,fontsize=9,ha='center',va='center',fontweight='bold')
        if i<len(steps)-1: ax.annotate('',xy=(steps[i+1][0]-0.8,y),xytext=(x+0.8,y),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    plt.tight_layout(); plt.savefig(f'{BASE}/rag/rag_pipeline.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] rag/rag_pipeline.jpg')

def rag_vs_finetuning():
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    # RAG
    axes[0].set_xlim(0,10); axes[0].set_ylim(0,8); axes[0].axis('off')
    axes[0].set_title('RAG', fontsize=14, fontweight='bold')
    rag_steps = [(5,7,'Query','#E3F2FD'),(5,5.5,'Retrieve\nDocs','#C8E6C9'),(5,4,'LLM +\nContext','#FFF3E0'),(5,2.5,'Answer','#F1F8E9')]
    for x,y,t,c in rag_steps:
        box = FancyBboxPatch((x-1.5,y-0.5),3,1,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        axes[0].add_patch(box); axes[0].text(x,y,t,fontsize=10,ha='center',va='center',fontweight='bold')
    for i in range(len(rag_steps)-1): axes[0].annotate('',xy=(5,rag_steps[i+1][1]+0.5),xytext=(5,rag_steps[i][1]-0.5),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    # Fine-tuning
    axes[1].set_xlim(0,10); axes[1].set_ylim(0,8); axes[1].axis('off')
    axes[1].set_title('Fine-Tuning', fontsize=14, fontweight='bold')
    ft_steps = [(5,7,'Training\nData','#E3F2FD'),(5,5.5,'Fine-tune\nLLM','#C8E6C9'),(5,4,'Custom\nModel','#FFF3E0'),(5,2.5,'Answer','#F1F8E9')]
    for x,y,t,c in ft_steps:
        box = FancyBboxPatch((x-1.5,y-0.5),3,1,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        axes[1].add_patch(box); axes[1].text(x,y,t,fontsize=10,ha='center',va='center',fontweight='bold')
    for i in range(len(ft_steps)-1): axes[1].annotate('',xy=(5,ft_steps[i+1][1]+0.5),xytext=(5,ft_steps[i][1]-0.5),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    plt.tight_layout(); plt.savefig(f'{BASE}/rag/rag_vs_finetuning.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] rag/rag_vs_finetuning.jpg')

# ============================================================
# BACKEND DIAGRAMS
# ============================================================

def backend_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis('off')
    ax.set_title('AI Backend Architecture', fontsize=16, fontweight='bold')
    components = [(7,7,'Frontend','#E3F2FD'),(7,5.5,'API Gateway','#C8E6C9'),
                  (7,4,'Backend API','#FFF3E0'),(3,2.5,'Database','#F3E5F5'),
                  (7,2.5,'ML Model','#FFEBEE'),(11,2.5,'Cache/Vector DB','#E0F7FA'),
                  (7,1,'Monitoring','#F1F8E9')]
    for x,y,t,c in components:
        box = FancyBboxPatch((x-1.5,y-0.5),3,1,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        ax.add_patch(box); ax.text(x,y,t,fontsize=11,ha='center',va='center',fontweight='bold')
    # Arrows
    ax.annotate('',xy=(7,6),xytext=(7,6.5),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    ax.annotate('',xy=(7,4.5),xytext=(7,5),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    ax.annotate('',xy=(4.5,3),xytext=(5.5,3.5),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    ax.annotate('',xy=(7,3),xytext=(7,3.5),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    ax.annotate('',xy=(9.5,3),xytext=(8.5,3.5),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    ax.annotate('',xy=(7,1.5),xytext=(7,2),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    plt.tight_layout(); plt.savefig(f'{BASE}/backend/backend_architecture.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] backend/backend_architecture.jpg')

def backend_api_flow():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0,14); ax.set_ylim(0,5); ax.axis('off')
    ax.set_title('API Request Flow', fontsize=16, fontweight='bold')
    steps = [(1,2.5,'Client\nRequest','#E3F2FD'),(3.5,2.5,'FastAPI\nEndpoint','#C8E6C9'),
             (6,2.5,'Validation\n(Pydantic)','#FFF3E0'),(8.5,2.5,'Service\nLayer','#F3E5F5'),
             (11,2.5,'Response','#E0F7FA')]
    for i,(x,y,t,c) in enumerate(steps):
        box = FancyBboxPatch((x-1,y-0.6),2,1.2,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',linewidth=2)
        ax.add_patch(box); ax.text(x,y,t,fontsize=10,ha='center',va='center',fontweight='bold')
        if i<len(steps)-1: ax.annotate('',xy=(steps[i+1][0]-1,y),xytext=(x+1,y),arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    plt.tight_layout(); plt.savefig(f'{BASE}/backend/api_flow.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] backend/api_flow.jpg')

# ============================================================
# MATH DIAGRAMS
# ============================================================

def math_sigmoid():
    fig, ax = plt.subplots(figsize=(10, 6))
    z = np.linspace(-10,10,200); sig = 1/(1+np.exp(-z))
    ax.plot(z,sig,'b-',lw=3,label='Sigmoid')
    ax.axhline(y=0.5,color='r',ls='--',lw=2,label='Threshold=0.5')
    ax.fill_between(z,sig,0.5,where=(sig>=0.5),alpha=0.2,color='green')
    ax.fill_between(z,sig,0.5,where=(sig<0.5),alpha=0.2,color='red')
    ax.set_xlabel('z',fontsize=12); ax.set_ylabel('sigma(z)',fontsize=12)
    ax.set_title('Sigmoid Function',fontsize=14,fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{BASE}/math/sigmoid.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] math/sigmoid.jpg')

def math_gradient_descent():
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.linspace(-3,3,100); y = np.linspace(-3,3,100); X,Y = np.meshgrid(x,y)
    Z = X**2 + 2*Y**2
    ax.contour(X,Y,Z,levels=20,cmap='viridis',alpha=0.8)
    w = np.array([-2.5,2.5]); path = [w.copy()]
    for _ in range(15): grad = np.array([2*w[0],4*w[1]]); w = w - 0.3*grad; path.append(w.copy())
    path = np.array(path)
    ax.plot(path[:,0],path[:,1],'r-o',lw=2,ms=6,label='GD Path')
    ax.plot(path[0,0],path[0,1],'go',ms=12,label='Start')
    ax.plot(path[-1,0],path[-1,1],'r*',ms=15,label='Minimum')
    ax.set_xlabel('w1'); ax.set_ylabel('w2')
    ax.set_title('Gradient Descent',fontsize=14,fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{BASE}/math/gradient_descent.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] math/gradient_descent.jpg')

def math_bias_variance():
    fig, ax = plt.subplots(figsize=(10, 6))
    mc = np.linspace(0,10,100)
    bias2 = 10*np.exp(-0.5*mc); var = 0.1*np.exp(0.3*mc); total = bias2+var+0.5
    ax.plot(mc,bias2,'b-',lw=2,label='Bias^2')
    ax.plot(mc,var,'r-',lw=2,label='Variance')
    ax.plot(mc,total,'k-',lw=3,label='Total Error')
    mi = np.argmin(total); ax.axvline(x=mc[mi],color='green',ls='--',alpha=0.7,label='Optimal')
    ax.fill_between(mc,0,3,alpha=0.1,color='blue'); ax.fill_between(mc,7,10,alpha=0.1,color='red')
    ax.text(1.5,7,'Underfitting',fontsize=12,color='blue',fontweight='bold')
    ax.text(8.5,7,'Overfitting',fontsize=12,color='red',fontweight='bold')
    ax.set_xlabel('Model Complexity'); ax.set_ylabel('Error')
    ax.set_title('Bias-Variance Tradeoff',fontsize=14,fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{BASE}/math/bias_variance.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] math/bias_variance.jpg')

def math_entropy_gini():
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    p = np.linspace(0.001,0.999,100)
    entropy = -p*np.log2(p)-(1-p)*np.log2(1-p)
    axes[0].plot(p,entropy,'b-',lw=3); axes[0].fill_between(p,entropy,alpha=0.2,color='blue')
    axes[0].set_xlabel('p(Class 1)'); axes[0].set_ylabel('Entropy')
    axes[0].set_title('Entropy',fontsize=13,fontweight='bold'); axes[0].grid(True,alpha=0.3)
    gini = 1-p**2-(1-p)**2
    axes[1].plot(p,gini,'r-',lw=3); axes[1].fill_between(p,gini,alpha=0.2,color='red')
    axes[1].set_xlabel('p(Class 1)'); axes[1].set_ylabel('Gini Impurity')
    axes[1].set_title('Gini Impurity',fontsize=13,fontweight='bold'); axes[1].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{BASE}/math/entropy_gini.jpg',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('[OK] math/entropy_gini.jpg')

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    ensure_dirs()
    print('='*60)
    print('GENERATING ALL COURSE DIAGRAMS')
    print('='*60)

    print('\n--- ML Diagrams ---')
    ml_workflow(); ml_algorithm_comparison(); ml_overfitting()
    ml_confusion_matrix(); ml_roc_curve(); ml_decision_boundary(); ml_cross_validation()

    print('\n--- DL Diagrams ---')
    dl_neural_network(); dl_activation_functions(); dl_backpropagation()
    dl_cnn_architecture(); dl_lstm_gates(); dl_gradient_descent()

    print('\n--- NLP Diagrams ---')
    nlp_pipeline(); nlp_word_embeddings(); nlp_sentiment()

    print('\n--- Transformer Diagrams ---')
    transformer_architecture(); transformer_attention()

    print('\n--- RAG Diagrams ---')
    rag_pipeline(); rag_vs_finetuning()

    print('\n--- Backend Diagrams ---')
    backend_architecture(); backend_api_flow()

    print('\n--- Math Diagrams ---')
    math_sigmoid(); math_gradient_descent(); math_bias_variance(); math_entropy_gini()

    print('\n' + '='*60)
    print('ALL DIAGRAMS GENERATED!')
    print('='*60)

    # Count
    count = 0
    for root, dirs, files in os.walk(BASE):
        for f in files:
            if f.endswith(('.jpg','.png')): count += 1
    print(f'Total diagrams: {count}')
