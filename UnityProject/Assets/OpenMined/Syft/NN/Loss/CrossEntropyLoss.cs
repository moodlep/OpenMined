using OpenMined.Syft.Tensor;
using OpenMined.Network.Controllers;

namespace OpenMined.Syft.Layer.Loss
{
    public class CrossEntropyLoss: Loss
    {

	    private int dim;
		
		public CrossEntropyLoss (SyftController controller, int dim)
		{
			init("crossentropyloss");

			this.dim = dim;
			
			#pragma warning disable 420
			id = System.Threading.Interlocked.Increment(ref nCreated);
			controller.addModel(this);

		}
        public override FloatTensor Forward(FloatTensor prediction, FloatTensor target)
		{
			// Note: prediction should be logits, basically pre-softmax. This method applies softmax first. 
			// TODO check shapes 

//			FloatTensor softmax = prediction.Softmax(this.dim);
//			FloatTensor output = ((target.Mul(softmax.Log())).Sum()).Neg();
//			return output;
			FloatTensor pred = prediction.Copy(autograd:true);
			FloatTensor output = (target.Mul(pred.LogSoftmax(this.dim))).Sum().Neg();
			FloatTensor loss = output.Copy(autograd: true); 
			//prediction.HookGraph(ref loss, creation_op: "copy", backward_op: "crossentropyloss", inline:false, tensor_inputs:new FloatTensor[]{target});
			prediction.HookGraph(ref loss, creation_op: "crossentropyloss", inline:false, tensor_inputs:new FloatTensor[]{target});
			target.Autograd = false;
			output.Autograd = false;
			pred.Autograd = false;

			return loss;
		}

		public override int getParameterCount(){return 0;}
		
	}
}

