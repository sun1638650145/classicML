//
//  exceptions.h
//  exceptions
//
//  Created by 孙瑞琦 on 2021/1/21.
//

#ifndef CLASSICML_BACKEND_CC_EXCEPTIONS_H_
#define CLASSICML_BACKEND_CC_EXCEPTIONS_H_

namespace exceptions {
class NotImplementedError : public std::exception {
    public:
        const char *what() const noexcept override {
            return "函数没有实现";
        }
};
}  // namespace exceptions

#endif /* CLASSICML_BACKEND_CC_EXCEPTIONS_H_ */