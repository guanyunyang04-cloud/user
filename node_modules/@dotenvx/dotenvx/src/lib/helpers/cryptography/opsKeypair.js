const Ops = require('../../extensions/ops')

async function opsKeypair (existingPublicKey, options = {}) {
  if (options.beforeOpsKeypair) await options.beforeOpsKeypair()

  let kp
  try {
    if (options.beforeOpsKeypair || options.afterOpsKeypair) {
      kp = await new Ops().keypair(existingPublicKey, { noSpinner: true })
    } else {
      kp = await new Ops().keypair(existingPublicKey)
    }
  } finally {
    if (options.afterOpsKeypair) await options.afterOpsKeypair()
  }

  const publicKey = kp.public_key
  const privateKey = kp.private_key

  return {
    publicKey,
    privateKey
  }
}

module.exports = opsKeypair
